#!/usr/bin/env python3
"""One locked 5-fold LUNG1 run: frozen Linear Cox, MLP, or SQA.

Neural heads use Lightning BF16 mixed precision and select the earliest epoch
with maximal validation Uno C. Cox selects its L2 penalty on the same held-out
validation split. Every test patient is predicted exactly once.
"""

from __future__ import annotations

import argparse
import json
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
from run_lung1_e2_encoder_bf16_lightning import aggregate, cox_loss, write_json  # noqa: E402
from run_lung1_e2_mlp import CoxMLP, EfronLoss, breslow_survival  # noqa: E402
from run_lung1_e2_sqa import SurvivalQueryAdapter, load_tokens, normalize_tokens  # noqa: E402
from run_lung1_frozen_cox_e1 import (  # noqa: E402
    ALPHAS, GRID, cohort, fit_cox, fold_metrics, inner_assignments,
    load_embeddings, locked_folds, predict as predict_cox, sha256, uno_c,
)

COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
FOLDS = ROOT / "configs/lung1_e1_folds.csv"
OUTPUT = ROOT / "outputs/lung1_bf16_selected"
MODELS = ("tangerine", "radjepa", "colipri", "ctclip", "dinov3", "pai", "hosny")
METHODS = ("linear", "mlp", "sqa")
EPOCHS = {"mlp": 100, "sqa": 40}
WEIGHT_DECAY = 1e-4


def predict_neural(net: torch.nn.Module, x: np.ndarray, device: torch.device,
                   method: str) -> np.ndarray:
    net.eval()
    batch = 16 if method == "sqa" else len(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return np.concatenate([
            net(torch.as_tensor(x[start:start + batch], device=device,
                                dtype=torch.float32)).float().cpu().numpy()
            for start in range(0, len(x), batch)
        ]).astype(np.float32)


class HeadFold(pl.LightningModule):
    def __init__(self, net: torch.nn.Module, method: str,
                 x_train: np.ndarray, t_train: np.ndarray, e_train: np.ndarray,
                 x_val: np.ndarray, t_val: np.ndarray, e_val: np.ndarray,
                 x_test: np.ndarray, t_test: np.ndarray, e_test: np.ndarray):
        super().__init__()
        self.net, self.method = net, method
        self.x_train, self.t_train, self.e_train = x_train, t_train, e_train
        self.x_val, self.t_val, self.e_val = x_val, t_val, e_val
        self.x_test, self.t_test, self.e_test = x_test, t_test, e_test
        self.objective = EfronLoss(t_train, e_train)
        self.best_epoch = 0
        self.best_val_uno_c = float("-inf")
        self.best_state: dict[str, torch.Tensor] = {}
        self.test_risk: np.ndarray | None = None
        self.test_survival: np.ndarray | None = None
        self.test_metrics: dict[str, float] = {}
        self.validation_history: list[dict[str, float]] = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=3e-4,
                                 weight_decay=WEIGHT_DECAY)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()

    def training_step(self, _batch, _batch_idx):
        x = torch.as_tensor(self.x_train, device=self.device, dtype=torch.float32)
        batch = 16 if self.method == "sqa" else len(x)
        risk = torch.cat([self.net(x[start:start + batch]).float()
                          for start in range(0, len(x), batch)])
        with torch.autocast("cuda", enabled=False):
            loss = self.objective(risk.float())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training Cox loss")
        self.log("train/optimization_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, _batch, _batch_idx):
        train_risk = predict_neural(self.net, self.x_train, self.device, self.method)
        val_risk = predict_neural(self.net, self.x_val, self.device, self.method)
        train_survival = breslow_survival(self.t_train, self.e_train,
                                          train_risk, train_risk)
        val_survival = breslow_survival(self.t_train, self.e_train,
                                        train_risk, val_risk)
        train_metric = fold_metrics(self.t_train, self.e_train,
                                    self.t_train, self.e_train, train_risk,
                                    train_survival)
        val_metric = fold_metrics(self.t_train, self.e_train,
                                  self.t_val, self.e_val, val_risk, val_survival)
        self.log("train/loss", cox_loss(self.t_train, self.e_train,
                                        train_risk, self.device), on_epoch=True)
        self.log("val/loss", cox_loss(self.t_val, self.e_val,
                                      val_risk, self.device), on_epoch=True)
        for key in ("uno_c_36m", "auc_2y", "ibs_3m_36m"):
            self.log(f"train/{key}", float(train_metric[key]), on_epoch=True)
            self.log(f"val/{key}", float(val_metric[key]), on_epoch=True)
        score = float(val_metric["uno_c_36m"])
        self.validation_history.append({"epoch": self.current_epoch + 1, "val_uno_c_36m": score})
        if np.isfinite(score) and score > self.best_val_uno_c:
            self.best_val_uno_c = score
            self.best_epoch = self.current_epoch + 1
            self.best_state = {name: p.detach().cpu().clone()
                               for name, p in self.net.named_parameters()}
        print(f"epoch={self.current_epoch + 1}/{self.trainer.max_epochs} "
              f"val_uno_c={score:.4f} best_epoch={self.best_epoch}", flush=True)

    def test_step(self, _batch, _batch_idx):
        train_risk = predict_neural(self.net, self.x_train, self.device, self.method)
        test_risk = predict_neural(self.net, self.x_test, self.device, self.method)
        test_survival = breslow_survival(self.t_train, self.e_train,
                                         train_risk, test_risk)
        self.test_metrics = fold_metrics(self.t_train, self.e_train,
                                         self.t_test, self.e_test,
                                         test_risk, test_survival)
        self.test_risk, self.test_survival = test_risk, test_survival
        self.log("test/loss", cox_loss(self.t_test, self.e_test,
                                       test_risk, self.device), on_epoch=True)
        for key, value in self.test_metrics.items():
            self.log(f"test/{key}", float(value), on_epoch=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the fixed neural epoch budget for a smoke run")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--validation-inner-fold", type=int, choices=(0, 1, 2), default=0,
                        help="Inner fold used for validation (nested-CV inner stages)")
    parser.add_argument("--fixed-trainval", action="store_true",
                        help="Refit on the whole outer training set and evaluate the final epoch")
    parser.add_argument("--fold-epochs", type=json.loads, default=None,
                        help='With --fixed-trainval: JSON mapping outer fold to epochs, e.g. {"0": 12}')
    parser.add_argument("--wandb-project", default="lung1-prognostic-transferability")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="offline")
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5:
        parser.error("max-folds must be 1 to 5")
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    if args.fixed_trainval:
        if args.method == "linear" or args.validation_inner_fold != 0 or not args.fold_epochs:
            parser.error("--fixed-trainval needs a neural method and --fold-epochs, without --validation-inner-fold")
        args.fold_epochs = {int(k): int(v) for k, v in args.fold_epochs.items()}
        if not all(v >= 1 for v in args.fold_epochs.values()):
            parser.error("fold epochs must be positive")
    elif args.fold_epochs is not None:
        parser.error("--fold-epochs requires --fixed-trainval")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 capable GPU is required")
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    inner = inner_assignments(data, folds)
    provenance: dict = {}
    if args.method == "sqa":
        features, provenance = load_tokens(args.model, data)
        provenance["kind"] = "cached_frozen_spatial_tokens"
    else:
        manifest = ROOT / "outputs/lung1_e1_embeddings" / args.model / "original_ct" / "embedding_manifest.csv"
        _, features, manifest_sha, matrix_sha = load_embeddings(f"{args.model}={manifest}", data)
        provenance = {"kind": "cached_frozen_global_embedding",
                      "manifest_sha256": manifest_sha, "matrix_sha256": matrix_sha}
    output = args.output_root / args.method / args.model
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": args.model, "method": args.method,
                "precision": "bf16-mixed-neural-forward-fp32-cox-loss",
                "outer_repetition": 0, "outer_folds": 5,
                "validation_inner_fold": 0,
                "selection": "maximum_validation_uno_c_36m_earliest_tie",
                "epochs": args.epochs or EPOCHS.get(args.method),
                "neural_settings": {"mlp_hidden": 32, "dropout": 0.1,
                                    "learning_rate": 3e-4, "weight_decay": WEIGHT_DECAY},
                "linear_components": 32, "linear_penalties": list(ALPHAS),
                "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "features": provenance}
    if args.validation_inner_fold != 0:
        protocol["validation_inner_fold"] = args.validation_inner_fold
    if args.fixed_trainval:
        protocol.update(validation_inner_fold=None, epochs=None,
                        fold_epochs={str(k): v for k, v in sorted(args.fold_epochs.items())},
                        selection="fixed_final_epoch_after_inner_cv_selection")
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a new output root")
    write_json(protocol_path, protocol)
    completed = {int(path.stem.split("_")[-1]) for path in (output / "folds").glob("fold_*.json")}
    aggregate(output, data)
    for fold in range(args.max_folds):
        if fold in completed:
            print(f"{args.method}/{args.model} fold={fold} already complete", flush=True)
            continue
        test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        if args.fixed_trainval and fold not in args.fold_epochs:
            continue
        val_ids = set() if args.fixed_trainval else set(
            inner.loc[(inner.repetition == 0) & (inner.fold == fold) &
                      (inner.inner_fold == args.validation_inner_fold), "patient_id"])
        test = data.patient_id.isin(test_ids).to_numpy()
        val = data.patient_id.isin(val_ids).to_numpy()
        train = ~(test | val)
        if (train & val).any() or (test & val).any() or train.sum() + val.sum() + test.sum() != len(data):
            raise ValueError("Invalid patient split")
        t = data.os_time_days.to_numpy(dtype=float)
        e = data.os_event.to_numpy(dtype=int)
        started = time.monotonic()
        fold_dir = output / "wandb" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.wandb_project,
                             name=f"5CV-BF16-{args.method}-{args.model}-fold{fold}",
                             group=f"lung1-5cv-bf16-{args.method}-{args.model}",
                             job_type="selected-checkpoint-cv", save_dir=str(fold_dir),
                             offline=args.wandb_mode == "offline", log_model=False)
        logger.log_hyperparams({"model": args.model, "method": args.method,
                                "fold": fold, "n_train": int(train.sum()),
                                "n_val": int(val.sum()), "n_test": int(test.sum()),
                                "precision": protocol["precision"]})
        try:
            if args.method == "linear":
                # Linear Cox has no epoch: choose the penalty on the held-out validation set.
                # FP32/64 Cox optimization is retained for numerical stability; there is no
                # neural forward pass on which BF16 autocast could act.
                scores = []
                candidates = []
                for penalty in ALPHAS:
                    fitted = fit_cox(features[train], t[train], e[train], penalty, 32)
                    val_risk, _ = predict_cox(fitted, features[val], GRID)
                    score = uno_c(t[train], e[train], t[val], e[val], val_risk)
                    scores.append(float(score))
                    candidates.append(fitted)
                    logger.experiment.log({"validation/penalty": penalty,
                                           "val/uno_c_36m": float(score)})
                if not np.isfinite(scores).all():
                    raise ValueError("Undefined validation Uno C")
                best = int(np.argmax(scores))
                fitted = candidates[best]
                risk, survival = predict_cox(fitted, features[test], GRID)
                metrics = fold_metrics(t[train], e[train], t[test], e[test], risk, survival)
                selected = {"selected_penalty": ALPHAS[best],
                            "selected_validation_uno_c_36m": scores[best]}
            else:
                from sklearn.preprocessing import StandardScaler
                pl.seed_everything(20261005 + fold, workers=True)
                if args.method == "mlp":
                    scaler = StandardScaler().fit(features[train])
                    x_train = scaler.transform(features[train]).astype(np.float32)
                    x_val = (scaler.transform(features[val]).astype(np.float32) if val.any()
                             else np.zeros((0, features.shape[1]), np.float32))
                    x_test = scaler.transform(features[test]).astype(np.float32)
                    net = CoxMLP(x_train.shape[1], 32, 0.1)
                else:
                    x_train, x_val = normalize_tokens(
                        features[train], features[val] if val.any() else features[train][:1])
                    if not val.any():
                        x_val = x_val[:0]
                    _, x_test = normalize_tokens(features[train], features[test])
                    net = SurvivalQueryAdapter(x_train.shape[-1], 0.1,
                                               tuple(provenance["saved_grid"]))
                module = HeadFold(net, args.method, x_train, t[train], e[train],
                                  x_val, t[val], e[val], x_test, t[test], e[test])
                epochs = args.fold_epochs[fold] if args.fixed_trainval else protocol["epochs"]
                trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
                                     max_epochs=epochs, logger=logger,
                                     limit_val_batches=0 if args.fixed_trainval else 1.0,
                                     enable_checkpointing=False, enable_progress_bar=False,
                                     enable_model_summary=False, num_sanity_val_steps=0,
                                     log_every_n_steps=1)
                trainer.fit(module)
                if args.fixed_trainval:
                    # Refit stage: no validation data; the final epoch is the model.
                    module.best_epoch = epochs
                    module.best_val_uno_c = float("nan")
                    module.best_state = {name: p.detach().cpu().clone()
                                         for name, p in module.net.named_parameters()}
                if not module.best_state:
                    raise RuntimeError("No finite validation Uno C")
                checkpoint = output / "checkpoints" / f"fold_{fold}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": module.best_epoch,
                            "validation_uno_c_36m": module.best_val_uno_c,
                            "state": module.best_state}, checkpoint)
                with torch.no_grad():
                    for name, parameter in module.net.named_parameters():
                        parameter.copy_(module.best_state[name].to(parameter.device))
                trainer.test(module, ckpt_path=None)
                risk, survival, metrics = module.test_risk, module.test_survival, module.test_metrics
                selected = {"selected_epoch": module.best_epoch,
                            "selected_validation_uno_c_36m": module.best_val_uno_c,
                            "trainable_parameters": sum(p.numel() for p in net.parameters())}
            row = {"model": args.model, "method": args.method, "repetition": 0,
                   "fold": fold, "n_train": int(train.sum()),
                   "n_validation": int(val.sum()), "n_test": int(test.sum()),
                   "elapsed_seconds": time.monotonic() - started,
                   **selected, **metrics}
            predictions = []
            for i, index in enumerate(np.flatnonzero(test)):
                prediction = {"model": args.model, "method": args.method,
                              "repetition": 0, "fold": fold,
                              "patient_id": data.patient_id.iloc[index],
                              "os_time_days": float(t[index]), "os_event": int(e[index]),
                              "risk": float(risk[i]),
                              "risk_36m": float(-np.log(np.clip(survival[i, -1], 1e-12, 1.0)))}
                prediction.update({f"survival_{j:02d}": float(value)
                                   for j, value in enumerate(survival[i])})
                predictions.append(prediction)
            payload = {"metrics": row, "predictions": predictions}
            if args.method != "linear" and not args.fixed_trainval:
                payload["validation_history"] = module.validation_history
            write_json(output / "folds" / f"fold_{fold}.json", payload)
            count = aggregate(output, data)
            logger.experiment.summary.update({"completed_folds": count, **selected,
                                               **{f"test/{key}": value for key, value in metrics.items()}})
            if count == 5:
                pooled = json.loads((output / "summary.json").read_text())["pooled"]
                logger.experiment.summary.update({f"pooled/{key}": value
                                                   for key, value in pooled.items()})
            print(f"{args.method}/{args.model} fold={fold} complete; "
                  f"test Uno C={metrics['uno_c_36m']:.3f}; completed={count}/5", flush=True)
        finally:
            logger.experiment.finish()


if __name__ == "__main__":
    main()
