#!/usr/bin/env python3
"""Run compact frozen-COLIPRI temporal-adapter ablations with Lightning/W&B."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lung_jepa.data.colipri_temporal import COLIPRITemporalDataset
from lung_jepa.models.temporal_adapter import MODEL_SPECS
from lung_jepa.models.temporal_losses import future_loss
from lung_jepa.models.temporal_module import TemporalFutureStateModule
from run_colipri_latent_world_model import HORIZONS, build_pairs

DEFAULT_MODELS = list(MODEL_SPECS)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); L.seed_everything(seed, workers=True)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config", type=Path, default=None)
    config_args, remaining = pre.parse_known_args()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--embedding-cache", type=Path, default=Path("outputs/colipri_decodability/mean/cache"))
    p.add_argument("--prognostic-labels", type=Path, default=Path("data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv"))
    p.add_argument("--diagnostic-manifest", type=Path, default=Path("outputs/colipri_decodability/experiment_scan_manifest.csv"))
    p.add_argument("--patient-splits", type=Path, default=Path("outputs/colipri_decodability/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/colipri_temporal_adapter_ablation"))
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); p.add_argument("--probe-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--latent-dim", type=int, default=128); p.add_argument("--adapter-hidden-dim", type=int, default=256); p.add_argument("--predictor-hidden-dim", type=int, default=256); p.add_argument("--predictor-blocks", type=int, default=2); p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=15); p.add_argument("--batch-size", type=int, default=128); p.add_argument("--num-workers", type=int, default=0); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss-alpha", type=float, default=1.0); p.add_argument("--lambda-sigreg", type=float, default=0.0); p.add_argument("--device", default="auto"); p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--no-wandb", action="store_true"); p.add_argument("--wandb-project", default="nlst-colipri-temporal-adapter"); p.add_argument("--wandb-entity", default=None); p.add_argument("--wandb-group", default="temporal-adapter-ablation")
    if config_args.config is not None:
        with config_args.config.open() as f:
            p.set_defaults(**(yaml.safe_load(f) or {}))
    args = p.parse_args(remaining); args.config = config_args.config
    return args


def device_from_arg(name: str) -> str:
    return name if name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def enrich_previous(pairs: pd.DataFrame, cache: Path) -> pd.DataFrame:
    metadata = pd.read_csv(cache / "embedding_metadata.csv", dtype=str); metadata["embedding_index"] = metadata["embedding_index"].astype(int)
    lookup = {(str(r.patient_id), str(r.screening_round)): int(r.embedding_index) for r in metadata.itertuples()}
    out = pairs.copy(); out["idx_prev"] = [lookup.get((str(pid), "T0"), -1) if typ == "T1_T2" else -1 for pid, typ in zip(out.patient_id, out.pair_type)]
    return out


def loaders(pairs: pd.DataFrame, embeddings: np.ndarray, args: argparse.Namespace):
    result = {}
    for split in ["train", "val", "test"]:
        ds = COLIPRITemporalDataset(pairs[pairs.split.eq(split)], embeddings)
        result[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train", num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return result


def latent_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target; cosine = np.sum(pred * target, axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1) + 1e-12)
    return {"mse": float(np.mean(diff ** 2)), "cosine": float(np.mean(cosine)), "l2": float(np.mean(np.linalg.norm(diff, axis=1))), "r2": float(r2_score(target, pred, multioutput="variance_weighted"))}


def collect(model, loader, device: str):
    # Export must follow dataset/label order, even for the shuffled training loader.
    loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0)
    model.eval().to(device); predictions, targets, states = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred, target, state = model(batch); predictions.append(pred.cpu().numpy()); targets.append(target.cpu().numpy()); states.append(state.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(states)


def smoke(model, device: str, dim: int) -> None:
    batch = {"source": torch.randn(2, dim), "target": torch.randn(2, dim), "previous": torch.randn(2, dim), "has_previous": torch.ones(2), "delta_t": torch.randn(2, 1)}
    model.to(device); batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        pred, target, state = model(batch); loss, parts = future_loss(pred, target, float(model.hparams.loss_alpha))
    print(json.dumps({"source_shape": list(batch["source"].shape), "state_shape": list(state.shape), "prediction_shape": list(pred.shape), "target_shape": list(target.shape), "loss": float(loss), "mse": float(parts["mse"]), "cosine_loss": float(parts["cosine_loss"])}, indent=2), flush=True)


def train_one(args, model_name: str, seed: int, loaders_by_split, input_dim: int, out_dir: Path, device: str):
    seed_all(seed)
    model = TemporalFutureStateModule(input_dim=input_dim, model_name=model_name, latent_dim=args.latent_dim, adapter_hidden_dim=args.adapter_hidden_dim, predictor_hidden_dim=args.predictor_hidden_dim, predictor_blocks=args.predictor_blocks, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, loss_alpha=args.loss_alpha, lambda_sigreg=args.lambda_sigreg)
    logger = CSVLogger(save_dir=str(out_dir), name=f"csv_{model_name}_{seed}")
    if not args.no_wandb:
        logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, name=f"{model_name}__seed{seed}", save_dir=str(out_dir / "wandb"), log_model=False)
        logger.experiment.config.update({"model_name": model_name, "seed": seed, "latent_dim": args.latent_dim, "loss_alpha": args.loss_alpha, "lambda_sigreg": args.lambda_sigreg})
    ckpt_dir = out_dir / "checkpoints" / model_name / f"seed{seed}"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best", monitor="val/loss", mode="min", save_top_k=1)
    trainer = L.Trainer(max_epochs=args.epochs, accelerator=device, devices=1, logger=logger, callbacks=[checkpoint, EarlyStopping(monitor="val/loss", mode="min", patience=args.patience), LearningRateMonitor(logging_interval="epoch")], enable_progress_bar=False, enable_model_summary=False, deterministic=True, default_root_dir=str(out_dir))
    trainer.fit(model, train_dataloaders=loaders_by_split["train"], val_dataloaders=loaders_by_split["val"])
    if checkpoint.best_model_path: model = TemporalFutureStateModule.load_from_checkpoint(checkpoint.best_model_path)
    test_results = trainer.test(model, dataloaders=loaders_by_split["test"], verbose=False)
    collected = {split: collect(model, loader, device) for split, loader in loaders_by_split.items()}
    np.savez(out_dir / "predictions" / f"{model_name}__seed{seed}.npz", **{f"{split}_{key}": value for split, values in collected.items() for key, value in zip(["pred", "target", "state"], values)})
    if not args.no_wandb: logger.experiment.finish()
    return collected, {"model": model_name, "seed": seed, "checkpoint": checkpoint.best_model_path, "test_loss": test_results[0].get("test/loss") if test_results else None}


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)), "mcc": float(matthews_corrcoef(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)), "specificity": float(tn / (tn + fp)) if tn + fp else np.nan}


def run_downstream(args, pairs: pd.DataFrame, all_features: dict, out_dir: Path) -> None:
    rows = []
    for run_name, features in all_features.items():
        for feature_name in ["z_t", "predicted_future", "predicted_delta", "concat"]:
            for label in [f"primary_future_cancer_{h}" for h in HORIZONS]:
                ys = {s: pd.to_numeric(pairs[pairs.split.eq(s)][label], errors="coerce").to_numpy(float) for s in ["train", "val", "test"]}; masks = {s: np.isfinite(ys[s]) for s in ys}
                if any(masks[s].sum() == 0 for s in masks) or len(np.unique(ys["train"][masks["train"]])) < 2: continue
                for probe_seed in args.probe_seeds:
                    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=probe_seed))
                    clf.fit(features["train"][feature_name][masks["train"]], ys["train"][masks["train"]].astype(int)); score = clf.predict_proba(features["test"][feature_name][masks["test"]])[:, 1]
                    rows.append({"run": run_name, "feature": feature_name, "label": label, "probe_seed": probe_seed, **classification_metrics(ys["test"][masks["test"]].astype(int), score)})
    pd.DataFrame(rows).to_csv(out_dir / "downstream_prognostic_metrics.csv", index=False)
    if rows: pd.DataFrame(rows).groupby(["run", "feature", "label"])[["auroc", "auprc", "mcc", "f1"]].agg(["mean", "std"]).to_csv(out_dir / "downstream_prognostic_summary.csv")


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "predictions").mkdir(exist_ok=True)
    device = device_from_arg(args.device)
    pair_args = SimpleNamespace(embedding_cache=args.embedding_cache, prognostic_labels=args.prognostic_labels, diagnostic_manifest=args.diagnostic_manifest, patient_splits=args.patient_splits)
    pairs, embeddings = build_pairs(pair_args); pairs = enrich_previous(pairs, args.embedding_cache); pairs.to_csv(args.output_dir / "pairs_manifest.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"pairs": len(pairs), "patients": int(pairs.patient_id.nunique()), "split_counts": pairs.split.value_counts().to_dict(), "models": args.models, "seeds": args.seeds, "device": device}, indent=2))
    loaders_by_split = loaders(pairs, embeddings, args)
    smoke(TemporalFutureStateModule(embeddings.shape[1], args.models[0], args.latent_dim, args.adapter_hidden_dim, args.predictor_hidden_dim, args.predictor_blocks, args.dropout, args.lr, args.weight_decay, args.loss_alpha, args.lambda_sigreg), device, embeddings.shape[1])
    if args.smoke_only: return
    all_latent, all_features, training_rows = [], {}, []
    for model_name in args.models:
        for seed in args.seeds:
            if model_name == "identity":
                collected = {split: (b["source"].numpy(), b["target"].numpy(), b["source"].numpy()) for split, loader in loaders_by_split.items() for b in []}
                collected = {}
                for split in ["train", "val", "test"]:
                    ds = loaders_by_split[split].dataset; source = np.stack([ds[i]["source"].numpy() for i in range(len(ds))]); target = np.stack([ds[i]["target"].numpy() for i in range(len(ds))]); collected[split] = (source, target, source)
                training_rows.append({"model": model_name, "seed": seed, "checkpoint": "identity", "test_loss": float(np.mean((collected["test"][0] - collected["test"][1]) ** 2))})
            else:
                collected, meta = train_one(args, model_name, seed, loaders_by_split, embeddings.shape[1], args.output_dir, device); training_rows.append(meta)
            run_name = f"{model_name}__seed{seed}"; all_features[run_name] = {}
            for split, (pred, target, state) in collected.items():
                all_features[run_name][split] = {"z_t": state, "predicted_future": pred, "predicted_delta": pred - state, "concat": np.concatenate([state, pred], axis=1)}
                all_latent.append({"run": run_name, "split": split, **latent_metrics(pred, target)})
    pd.DataFrame(training_rows).to_csv(args.output_dir / "training_per_seed_metrics.csv", index=False); pd.DataFrame(all_latent).to_csv(args.output_dir / "latent_per_seed_metrics.csv", index=False)
    run_downstream(args, pairs, all_features, args.output_dir)
    (args.output_dir / "README.md").write_text("Per-seed Lightning checkpoints, predictions, latent metrics, and downstream prognostic metrics are stored here.\n")
    print(f"Done. Results: {args.output_dir}", flush=True)


if __name__ == "__main__": main()
