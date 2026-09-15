#!/usr/bin/env python3
"""Evaluate clinical downstream tasks from latent predictor outputs.

The script loads a trained latent predictor checkpoint, generates predicted
future latents for train/validation/test transitions, and trains simple
regularized logistic-regression classifiers on clinically relevant labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lung_jepa.data.latent_transitions import LatentTransitionDataModule, LatentTransitionDataset
from lung_jepa.models.latent_predictor import LatentPredictorModule


FEATURE_SETS = {
    "source_z_t": "source",
    "pred_z_t1": "pred",
    "target_z_t1": "target",
    "pred_delta": "pred_delta",
    "true_delta": "true_delta",
}


def pool_if_tokens(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array.mean(axis=1)
    return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--transitions",
        type=Path,
        default=Path("/ssd2/domenico/datasets/nlst/processed/nlst/pilot_500_balanced/radjepa_transitions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=["cancer_label"],
        help="Binary outcome columns to evaluate if present in the transitions table.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-clinical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for latent predictor inference: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["direct", "residual"],
        default=None,
        help="Optional metadata only; checkpoint hparams define model behavior.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print one progress line every N inference batches; set 0 to disable.",
    )
    return parser.parse_args()


def device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def iter_split_batches(datamodule: LatentTransitionDataModule, split: str) -> Iterable[dict]:
    if split == "train":
        table = datamodule.train_df
    elif split == "val":
        table = datamodule.val_df
    elif split == "test":
        table = datamodule.test_df
    else:
        raise ValueError(f"Unknown split: {split}")
    if table is None or len(table) == 0:
        return []
    return DataLoader(
        LatentTransitionDataset(table, datamodule.clinical_columns, datamodule.use_delta),
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=datamodule.num_workers,
        pin_memory=False,
    )


def collect_split(
    model: LatentPredictorModule,
    datamodule: LatentTransitionDataModule,
    split: str,
    device: torch.device,
    progress_every: int = 25,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    loader = iter_split_batches(datamodule, split)
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    total_items = len(getattr(datamodule, f"{split}_df")) if getattr(datamodule, f"{split}_df") is not None else 0
    print(
        f"[downstream] collect split={split} n={total_items} batches={total_batches} device={device}",
        flush=True,
    )
    source_parts = []
    target_parts = []
    pred_parts = []
    started = time.time()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            source = batch["source"].float().to(device)
            target = batch["target"].float().to(device)
            clinical = batch["clinical"].float().to(device)
            delta = batch["delta"].float().to(device)
            pred = model(source, clinical, delta)
            source_parts.append(source.cpu().numpy())
            target_parts.append(target.cpu().numpy())
            pred_parts.append(model._pool_if_tokens(pred).cpu().numpy())
            if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - started
                print(
                    f"[downstream] split={split} batch={batch_idx}/{total_batches} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    source = np.concatenate(source_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    pred = np.concatenate(pred_parts, axis=0)
    source_pooled = pool_if_tokens(source)
    target_pooled = pool_if_tokens(target)
    features = {
        "source_z_t": source_pooled,
        "pred_z_t1": pred,
        "target_z_t1": target_pooled,
        "pred_delta": pred - source_pooled,
        "true_delta": target_pooled - source_pooled,
    }
    print(f"[downstream] collected split={split} feature_shape={pred.shape}", flush=True)
    table = getattr(datamodule, f"{split}_df")
    if table is None:
        raise RuntimeError(f"Missing {split}_df.")
    return features, table.reset_index(drop=True)


def clean_binary_labels(values: pd.Series) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce")
    valid = y.notna()
    unique = sorted(set(y[valid].astype(int).tolist()))
    if not set(unique).issubset({0, 1}):
        raise ValueError(f"Outcome is not binary 0/1: {unique}")
    return y.to_numpy(dtype=float)


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }


def fit_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int,
    seed: int,
) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    clf.fit(x_train, y_train)
    val_score = clf.predict_proba(x_val)[:, 1]
    test_score = clf.predict_proba(x_test)[:, 1]
    out = {}
    out.update({f"val_{k}": v for k, v in evaluate_predictions(y_val, val_score).items()})
    out.update({f"test_{k}": v for k, v in evaluate_predictions(y_test, test_score).items()})
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[downstream] output_dir={args.output_dir}", flush=True)
    print(f"[downstream] checkpoint={args.checkpoint}", flush=True)
    print(f"[downstream] transitions={args.transitions}", flush=True)

    datamodule = LatentTransitionDataModule(
        transitions_path=args.transitions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        use_clinical=args.use_clinical,
        use_delta=args.use_delta,
    )
    datamodule.setup("test")
    device = device_from_arg(args.device)
    print(f"[downstream] device={device}", flush=True)
    model = LatentPredictorModule.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.to(device)
    checkpoint_hparams = {
        key: value
        for key, value in dict(model.hparams).items()
        if key
        in {
            "model_arch",
            "prediction_mode",
            "embedding_dim",
            "clinical_dim",
            "delta_dim",
            "max_tokens",
            "token_loss_weight",
        }
    }

    split_features = {}
    split_tables = {}
    for split in ["train", "val", "test"]:
        split_features[split], split_tables[split] = collect_split(
            model,
            datamodule,
            split,
            device,
            progress_every=args.progress_every,
        )

    rows = []
    skipped = []
    for outcome in args.outcomes:
        if outcome not in split_tables["train"].columns:
            skipped.append({"outcome": outcome, "reason": "column not found"})
            continue
        try:
            y_train_all = clean_binary_labels(split_tables["train"][outcome])
            y_val_all = clean_binary_labels(split_tables["val"][outcome])
            y_test_all = clean_binary_labels(split_tables["test"][outcome])
        except ValueError as exc:
            skipped.append({"outcome": outcome, "reason": str(exc)})
            continue

        train_mask = np.isfinite(y_train_all)
        val_mask = np.isfinite(y_val_all)
        test_mask = np.isfinite(y_test_all)
        y_train = y_train_all[train_mask].astype(int)
        y_val = y_val_all[val_mask].astype(int)
        y_test = y_test_all[test_mask].astype(int)

        if len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            skipped.append({"outcome": outcome, "reason": "one split has fewer than two classes"})
            continue

        for feature_name in FEATURE_SETS:
            print(f"[downstream] fit outcome={outcome} feature_set={feature_name}", flush=True)
            result = fit_and_eval(
                split_features["train"][feature_name][train_mask],
                y_train,
                split_features["val"][feature_name][val_mask],
                y_val,
                split_features["test"][feature_name][test_mask],
                y_test,
                max_iter=args.max_iter,
                seed=args.seed,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "feature_set": feature_name,
                    "checkpoint": str(args.checkpoint),
                    "use_clinical_in_predictor": args.use_clinical,
                    "use_delta_in_predictor": args.use_delta,
                    **result,
                }
            )

    results = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    results_path = args.output_dir / "downstream_results.csv"
    skipped_path = args.output_dir / "downstream_skipped.csv"
    metadata_path = args.output_dir / "downstream_metadata.json"
    xlsx_path = args.output_dir / "downstream_results.xlsx"

    results.to_csv(results_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    with metadata_path.open("w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "transitions": str(args.transitions),
                "outcomes": args.outcomes,
                "feature_sets": list(FEATURE_SETS),
                "split_counts": {
                    "train": len(split_tables["train"]),
                    "val": len(split_tables["val"]),
                    "test": len(split_tables["test"]),
                },
                "use_clinical": args.use_clinical,
                "use_delta": args.use_delta,
                "prediction_mode": args.prediction_mode,
                "checkpoint_hparams": checkpoint_hparams,
            },
            f,
            indent=2,
        )
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="results", index=False)
            skipped_df.to_excel(writer, sheet_name="skipped", index=False)
    except ImportError:
        print("openpyxl is not installed; wrote CSV only.")

    print(f"Done. results={results_path}")


if __name__ == "__main__":
    main()
