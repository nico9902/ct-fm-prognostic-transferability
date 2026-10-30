#!/usr/bin/env python3
"""Single-repeat LUNG1 E2 encoder adaptation with Lightning BF16 mixed precision.

Each locked outer fold has a fixed validation split (inner fold zero).
The outer test fold is evaluated once with the best validation Uno C epoch.
With --fixed-trainval, the entire outer training split is used and only the
predeclared final epoch is evaluated, without validation checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_lung1_e1_embeddings import CHECKPOINTS, load_encoder  # noqa: E402
from run_lung1_e2_encoder_adaptation import (  # noqa: E402
    AdaptedRisk, METHODS, MODELS, SETTINGS, load_inputs,
)
from run_lung1_e2_mlp import EfronLoss, breslow_survival  # noqa: E402
from run_lung1_frozen_cox_e1 import (  # noqa: E402
    GRID, HORIZONS, cohort, cumulative_auc, fold_metrics, inner_assignments,
    locked_folds, sha256,
)

COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
FOLDS = ROOT / "configs/lung1_e1_folds.csv"
DEFAULT_OUTPUT = ROOT / "outputs/lung1_bf16_selected"
EPOCH_METRICS = ("uno_c_36m", "auc_2y", "ibs_3m_36m")


def precision_options(precision: str) -> dict:
    if precision == "16-mixed":
        from lightning.pytorch.plugins.precision import MixedPrecision
        return {"plugins": [MixedPrecision("16-mixed", "cuda",
                    scaler=torch.amp.GradScaler("cuda", init_scale=128.0))]}
    return {"precision": precision}


def risk_one(net: AdaptedRisk, path: Path, device: torch.device) -> torch.Tensor:
    """One cached CT; autocast selects BF16/FP16 according to the trainer."""
    volume = np.load(path, allow_pickle=False, mmap_mode="r")
    image = torch.from_numpy(np.asarray(volume, dtype=np.float16).copy())
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
    image = image.to(device=device, dtype=dtype)[None, None]
    return net(image)[0].float()


def risk_all(net: AdaptedRisk, paths: list[Path], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return np.asarray([float(risk_one(net, path, device)) for path in paths], dtype=np.float32)


def cox_loss(time_: np.ndarray, event: np.ndarray, risk: np.ndarray | torch.Tensor,
             device: torch.device) -> torch.Tensor:
    score = torch.as_tensor(risk, device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", enabled=False):
        return EfronLoss(time_, event)(score)


def dummy_loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


class FoldModel(pl.LightningModule):
    def __init__(self, net: AdaptedRisk, method: str,
                 train_paths: list[Path], train_time: np.ndarray, train_event: np.ndarray,
                 val_paths: list[Path], val_time: np.ndarray, val_event: np.ndarray,
                 test_paths: list[Path], test_time: np.ndarray, test_event: np.ndarray):
        super().__init__()
        self.net = net
        self.method = method
        self.train_paths, self.train_time, self.train_event = train_paths, train_time, train_event
        self.val_paths, self.val_time, self.val_event = val_paths, val_time, val_event
        self.test_paths, self.test_time, self.test_event = test_paths, test_time, test_event
        self.automatic_optimization = False
        self.last_train_loss = float("nan")
        self.last_val_loss = float("nan")
        self.last_train_metrics: dict[str, float] = {}
        self.last_val_metrics: dict[str, float] = {}
        self.test_metrics: dict[str, float] = {}
        self.test_risk: np.ndarray | None = None
        self.test_survival: np.ndarray | None = None
        self.best_val_uno_c = float("-inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] = {}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)

    def configure_optimizers(self):
        config = SETTINGS[self.method]
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                 lr=config["lr"], weight_decay=config["weight_decay"])

    def train_dataloader(self):
        return dummy_loader()

    def val_dataloader(self):
        return dummy_loader()

    def test_dataloader(self):
        return dummy_loader()

    def training_step(self, _batch, _batch_idx):
        # Eval mode makes both risk passes deterministic while retaining gradients.
        self.net.eval()
        detached = risk_all(self.net, self.train_paths, self.device)
        risk = torch.tensor(detached, device=self.device, dtype=torch.float32,
                            requires_grad=True)
        loss = cox_loss(self.train_time, self.train_event, risk, self.device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training Cox loss")
        derivative = torch.autograd.grad(loss, risk)[0].detach()
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        for i, path in enumerate(self.train_paths):
            self.manual_backward(risk_one(self.net, path, self.device) * derivative[i])
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        previous_scale = scaler.get_scale() if scaler is not None else None
        optimizer.step()
        if scaler is not None and scaler.get_scale() < previous_scale:
            raise FloatingPointError("FP16 gradient overflow skipped an update; use FP32 in a new output root")
        print(f"epoch={self.current_epoch + 1}/{self.trainer.max_epochs} "
              f"train_loss={float(loss.detach()):.6f}", flush=True)
        self.last_train_loss = float(loss.detach())
        self.log("train/optimization_loss", loss.detach(), on_step=False, on_epoch=True)
        return loss.detach()

    def validation_step(self, _batch, _batch_idx):
        train_risk = risk_all(self.net, self.train_paths, self.device)
        val_risk = risk_all(self.net, self.val_paths, self.device)
        train_loss = cox_loss(self.train_time, self.train_event, train_risk, self.device)
        val_loss = cox_loss(self.val_time, self.val_event, val_risk, self.device)
        train_survival = breslow_survival(self.train_time, self.train_event,
                                          train_risk, train_risk)
        val_survival = breslow_survival(self.train_time, self.train_event,
                                        train_risk, val_risk)
        train_metrics = fold_metrics(self.train_time, self.train_event,
                                     self.train_time, self.train_event, train_risk,
                                     train_survival)
        val_metrics = fold_metrics(self.train_time, self.train_event,
                                   self.val_time, self.val_event, val_risk, val_survival)
        self.last_train_loss = float(train_loss)
        self.last_val_loss = float(val_loss)
        self.last_train_metrics = train_metrics
        self.last_val_metrics = val_metrics
        selection_score = float(val_metrics["uno_c_36m"])
        if np.isfinite(selection_score) and selection_score > self.best_val_uno_c:
            self.best_val_uno_c = selection_score
            self.best_epoch = self.current_epoch + 1
            self.best_state = {name: param.detach().cpu().clone()
                               for name, param in self.net.named_parameters()
                               if param.requires_grad}
        self.log("train/loss", train_loss, on_step=False, on_epoch=True)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True)
        for key in EPOCH_METRICS:
            self.log(f"train/{key}", float(train_metrics[key]), on_step=False, on_epoch=True)
            self.log(f"val/{key}", float(val_metrics[key]), on_step=False, on_epoch=True)
        print(f"epoch={self.current_epoch + 1}/{self.trainer.max_epochs} "
              f"train_loss={self.last_train_loss:.4f} val_loss={self.last_val_loss:.4f} "
              f"train_uno_c={train_metrics['uno_c_36m']:.3f} "
              f"val_uno_c={val_metrics['uno_c_36m']:.3f}", flush=True)

    def test_step(self, _batch, _batch_idx):
        train_risk = risk_all(self.net, self.train_paths, self.device)
        test_risk = risk_all(self.net, self.test_paths, self.device)
        test_survival = breslow_survival(self.train_time, self.train_event,
                                         train_risk, test_risk)
        test_loss = cox_loss(self.test_time, self.test_event, test_risk, self.device)
        reference_time, reference_event = getattr(
            self, "metric_reference", (self.train_time, self.train_event))
        self.test_metrics = fold_metrics(reference_time, reference_event,
                                         self.test_time, self.test_event,
                                         test_risk, test_survival)
        self.test_risk, self.test_survival = test_risk, test_survival
        self.log("test/loss", test_loss, on_step=False, on_epoch=True)
        for key, value in self.test_metrics.items():
            self.log(f"test/{key}", float(value), on_step=False, on_epoch=True)
        print(f"test_uno_c={self.test_metrics['uno_c_36m']:.3f}", flush=True)


def write_json(path: Path, value: dict) -> None:
    def normalize(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True,
                                    default=normalize) + "\n")
    temporary.replace(path)


def aggregate(output: Path, data: pd.DataFrame) -> int:
    files = sorted((output / "folds").glob("fold_*.json"))
    if not files:
        return 0
    records = [json.loads(path.read_text()) for path in files]
    metrics = pd.DataFrame([record["metrics"] for record in records])
    predictions = pd.DataFrame([prediction for record in records
                                for prediction in record["predictions"]])
    if metrics.fold.duplicated().any() or predictions.patient_id.duplicated().any():
        raise ValueError("Duplicate fold or out-of-fold patient prediction")
    metrics.sort_values("fold").to_csv(output / "fold_metrics.csv", index=False)
    predictions.sort_values("patient_id").to_csv(output / "out_of_fold_predictions.csv", index=False)
    if len(files) == 5:
        if set(predictions.patient_id) != set(data.patient_id) or len(predictions) != len(data):
            raise ValueError("Pooled predictions must cover all 416 patients once")
        matched = predictions.set_index("patient_id").loc[data.patient_id]
        survival = matched[[f"survival_{i:02d}" for i in range(len(GRID))]].to_numpy(dtype=float)
        time_ = data.os_time_days.to_numpy(dtype=float)
        event = data.os_event.to_numpy(dtype=int)
        # Separate Cox fits can shift their raw log-risk scores. Absolute
        # cumulative hazards from each fold's training-only baseline are pooled.
        risk_36m = -np.log(np.clip(survival[:, -1], 1e-12, 1.0))
        pooled = fold_metrics(time_, event, time_, event, risk_36m, survival)
        for horizon in HORIZONS:
            column = int(np.argmin(abs(GRID - horizon)))
            horizon_risk = -np.log(np.clip(survival[:, column], 1e-12, 1.0))
            pooled[f"auc_{int(round(horizon / 365.25))}y"] = cumulative_auc(
                time_, event, time_, event, horizon_risk, horizon)
        summary = {"completed_folds": 5, "patients": len(data), "pooled": pooled,
                   "pooled_risk": "negative_log_predicted_survival_at_36_months",
                   "mean_fold": {key: float(metrics[key].mean()) for key in pooled}}
        write_json(output / "summary.json", summary)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--precision", choices=("bf16-mixed", "16-mixed", "32-true"), default="bf16-mixed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validation-inner-fold", type=int, choices=(0, 1, 2), default=0,
                        help="Inner fold used for validation (nested-CV inner stages)")
    parser.add_argument("--fixed-trainval", action="store_true",
                        help="Train on the entire outer training set; evaluate the final epoch")
    parser.add_argument("--budget-pct", type=int, choices=(10, 25, 50, 100))
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wandb-project", default="lung1-prognostic-transferability")
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    args = parser.parse_args()
    if args.learning_rate is not None:
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("learning-rate must be finite and positive")
        SETTINGS[args.method] = {**SETTINGS[args.method], "lr": args.learning_rate}
    if args.fixed_trainval and args.validation_inner_fold != 0:
        parser.error("--validation-inner-fold has no effect with --fixed-trainval")
    if args.epochs < 1 or not 1 <= args.max_folds <= 5:
        parser.error("epochs must be positive and max-folds must be 1 to 5")
    if args.budget_pct is not None and (not args.fixed_trainval or not 0 <= args.draw < 10
                                       or (args.budget_pct == 100 and args.draw != 0)):
        parser.error("E3 requires fixed-trainval, draw 0..9 (only draw 0 at 100%)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requires Ampere or newer; use --precision 16-mixed on V100")

    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    inner = inner_assignments(data, folds)
    paths, input_provenance = load_inputs(args.model, data)
    output = args.output_root / args.method / args.model
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": args.model, "method": args.method, "epochs": args.epochs,
                "precision": args.precision, "framework": "lightning",
                "outer_repetition": 0, "outer_folds": 5,
                "validation_inner_fold": args.validation_inner_fold,
                "training_fraction_of_outer_train": "two_thirds",
                "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "checkpoint_sha256": sha256(CHECKPOINTS[args.model]),
                "inputs": input_provenance, "settings": SETTINGS[args.method],
                "loss": "efron_cox_partial_log_likelihood",
                "selection": "earliest_epoch_with_max_validation_uno_c_36m",
                "risk_set_loss_precision": "fp32"}
    if args.precision == "16-mixed":
        protocol["gradient_scaler_initial_scale"] = 128.0
    if args.fixed_trainval:
        protocol.update(validation_inner_fold=None,
                        training_fraction_of_outer_train="all",
                        selection="fixed_final_epoch_without_validation")
    subsets = None
    if args.budget_pct is not None:
        from run_lung1_e3_frozen_linear import make_subsets
        subsets, table = make_subsets(data, folds)
        saved = ROOT / "outputs/lung1_e3_frozen_linear/subsets.csv"
        if not pd.read_csv(saved).equals(table):
            raise ValueError("E3 subsets differ from frozen experiment")
        table.to_csv(output / "subsets.csv", index=False)
        protocol.update(experiment="E3-exploratory-LoRA", budget_pct=args.budget_pct,
                        draw=args.draw, subsets_sha256=sha256(saved),
                        metric_censoring="entire_outer_training_set",
                        training_fraction_of_outer_train=args.budget_pct / 100)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a new output root")
    write_json(protocol_path, protocol)
    completed = {int(path.stem.split("_")[-1]) for path in (output / "folds").glob("fold_*.json")}
    aggregate(output, data)
    for fold in range(args.max_folds):
        if fold in completed:
            print(f"{args.method}/{args.model} fold={fold}: already complete", flush=True)
            continue
        outer_test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        inner_val_ids = set(inner.loc[(inner.repetition == 0) & (inner.fold == fold) &
                                      (inner.inner_fold == args.validation_inner_fold), "patient_id"])
        test_mask = data.patient_id.isin(outer_test_ids).to_numpy()
        val_mask = data.patient_id.isin(inner_val_ids).to_numpy()
        if args.fixed_trainval:
            val_mask[:] = False
        train_mask = ~(test_mask | val_mask)
        if not (train_mask.sum() + val_mask.sum() + test_mask.sum() == len(data)
                and not (test_mask & val_mask).any()):
            raise ValueError("Invalid train/validation/test split")
        outer_train_mask = ~test_mask
        if subsets is not None:
            train_mask[:] = False
            train_mask[subsets[(fold, args.draw, args.budget_pct)]] = True
            if (train_mask & test_mask).any():
                raise ValueError("E3 training overlaps test")
        def subset(mask):
            idx = np.flatnonzero(mask)
            return ([paths[i] for i in idx], data.os_time_days.to_numpy(dtype=float)[idx],
                    data.os_event.to_numpy(dtype=int)[idx], idx)
        train_paths, train_time, train_event, _ = subset(train_mask)
        val_paths, val_time, val_event, _ = subset(val_mask)
        test_paths, test_time, test_event, test_idx = subset(test_mask)
        pl.seed_everything(20261005 + fold, workers=True)
        encoder, _, _ = load_encoder(args.model, torch.device("cpu"), CHECKPOINTS[args.model])
        net = AdaptedRisk(args.model, encoder, args.method)
        module = FoldModel(net, args.method, train_paths, train_time, train_event,
                           val_paths, val_time, val_event, test_paths, test_time, test_event)
        if subsets is not None:
            module.metric_reference = (data.os_time_days.to_numpy(dtype=float)[outer_train_mask],
                                       data.os_event.to_numpy(dtype=int)[outer_train_mask])
        fold_dir = output / "wandb" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.wandb_project,
                             name=f"E2-BF16-{args.method}-{args.model}-fold{fold}",
                             group=f"lung1-e2-bf16-{args.method}-{args.model}",
                             job_type="encoder-adaptation-cv", save_dir=str(fold_dir),
                             offline=args.wandb_mode == "offline", log_model=False)
        logger.log_hyperparams({"model": args.model, "method": args.method, "fold": fold,
                                "epochs": args.epochs, "precision": args.precision,
                                "n_train": len(train_paths), "n_val": len(val_paths),
                                "n_test": len(test_paths),
                                "validation_inner_fold": None if args.fixed_trainval else args.validation_inner_fold,
                                "learning_rate": SETTINGS[args.method]["lr"],
                                "selection": protocol["selection"]})
        trainer = pl.Trainer(accelerator="gpu", devices=1, **precision_options(args.precision),
                             max_epochs=args.epochs, logger=logger, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             limit_val_batches=0 if args.fixed_trainval else 1.0,
                             num_sanity_val_steps=0, log_every_n_steps=1)
        started = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            trainer.fit(module)
            if args.fixed_trainval:
                module.best_epoch = args.epochs
                module.best_val_uno_c = float("nan")
                module.best_state = {name: param.detach().cpu().clone()
                                     for name, param in module.net.named_parameters()
                                     if param.requires_grad}
            if not module.best_state:
                raise RuntimeError("No finite validation Uno C; cannot select checkpoint")
            checkpoint_path = output / "checkpoints" / f"fold_{fold}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": module.best_epoch,
                        "validation_uno_c_36m": module.best_val_uno_c,
                        "trainable_state": module.best_state}, checkpoint_path)
            with torch.no_grad():
                for name, parameter in module.net.named_parameters():
                    if name in module.best_state:
                        parameter.copy_(module.best_state[name].to(parameter.device))
            trainer.test(module, ckpt_path=None)
            if module.test_risk is None or module.test_survival is None:
                raise RuntimeError("Lightning test pass produced no predictions")
            parameters = [p for p in module.parameters() if p.requires_grad]
            row = {"model": args.model, "method": args.method, "repetition": 0,
                   "fold": fold, "precision": args.precision, "n_train": len(train_paths),
                   "n_validation": len(val_paths), "n_test": len(test_paths),
                   "trainable_parameters": sum(p.numel() for p in parameters),
                   "total_parameters": sum(p.numel() for p in module.parameters()),
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                   "elapsed_seconds": time.monotonic() - started,
                   "selected_epoch": module.best_epoch,
                   "selected_validation_uno_c_36m": module.best_val_uno_c,
                   "train_loss": module.last_train_loss,
                   "validation_loss": module.last_val_loss, **module.test_metrics}
            predictions = []
            for i, index in enumerate(test_idx):
                prediction = {"model": args.model, "method": args.method,
                              "repetition": 0, "fold": fold,
                              "patient_id": data.patient_id.iloc[index],
                              "os_time_days": test_time[i], "os_event": test_event[i],
                              "risk": float(module.test_risk[i]),
                              "risk_36m": float(-np.log(np.clip(
                                  module.test_survival[i, -1], 1e-12, 1.0)))}
                prediction.update({f"survival_{j:02d}": float(value)
                                   for j, value in enumerate(module.test_survival[i])})
                predictions.append(prediction)
            write_json(output / "folds" / f"fold_{fold}.json",
                       {"metrics": row, "predictions": predictions,
                        "train_patient_ids": data.loc[train_mask, "patient_id"].tolist()})
            count = aggregate(output, data)
            logger.experiment.summary.update({"completed_folds": count,
                                               "selected_epoch": module.best_epoch,
                                               "selected_validation_uno_c_36m": module.best_val_uno_c,
                                               "test/uno_c_36m": row["uno_c_36m"]})
            if count == 5:
                pooled = json.loads((output / "summary.json").read_text())["pooled"]
                logger.experiment.summary.update({f"pooled/{key}": value
                                                   for key, value in pooled.items()})
            print(f"{args.method}/{args.model} fold={fold} complete; "
                  f"test Uno C={row['uno_c_36m']:.3f}; completed={count}/5", flush=True)
        finally:
            logger.experiment.finish()
            del trainer, module, net, encoder
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
