from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from lung_jepa.data.latent_transitions import LatentTransitionDataModule
from lung_jepa.models.latent_predictor import LatentPredictorModule


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Train latent JEPA world-model predictor.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config with argument defaults.")
    parser.add_argument("--transitions", type=Path, default=Path("data/processed/nlst/radjepa_transitions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints/latent_predictor"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model-arch",
        choices=["mlp", "transformer_cls", "transformer_token_cls", "transformer_token"],
        default="mlp",
    )
    parser.add_argument("--prediction-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--transformer-dim", type=int, default=384)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-mlp-dim", type=int, default=1536)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument(
        "--token-loss-weight",
        type=float,
        default=1.0,
        help="Weight for token-level loss when the predictor outputs a token sequence.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--use-clinical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run test evaluation on the best checkpoint after training.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log metrics and config to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="lung-jepa-world-model")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-offline", action=argparse.BooleanOptionalAction, default=False)

    if config_args.config is not None:
        with config_args.config.open() as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in config.items()})
    args = parser.parse_args(remaining)
    args.config = config_args.config
    args.transitions = Path(args.transitions)
    args.output_dir = Path(args.output_dir)
    return args


def build_logger(args: argparse.Namespace):
    if args.wandb:
        wandb_dir = args.output_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(wandb_dir))
        os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_dir / "cache"))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_dir / "config"))
        if args.wandb_offline:
            os.environ.setdefault("WANDB_MODE", "offline")
        return WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            save_dir=str(args.output_dir),
            offline=args.wandb_offline,
            log_model=False,
        )
    return CSVLogger(save_dir=str(args.output_dir), name="csv_logs")


def _as_float(value: Any) -> float | str | int | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def flatten_test_results(test_results: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not test_results:
        return {}
    return {key: _as_float(value) for key, value in test_results[0].items()}


def write_experiment_report(
    output_dir: Path,
    metadata: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    split_counts = metadata["split_counts"]
    prediction_outputs = metadata.get("prediction_outputs") or {}
    summary_rows = [
        {"field": "best_model_path", "value": metadata["best_model_path"]},
        {"field": "best_model_score", "value": metadata["best_model_score"]},
        {"field": "embedding_dim", "value": metadata["embedding_dim"]},
        {"field": "clinical_dim", "value": metadata["clinical_dim"]},
        {"field": "delta_dim", "value": metadata["delta_dim"]},
        {"field": "model_arch", "value": metadata.get("model_arch", "mlp")},
        {"field": "prediction_mode", "value": metadata.get("prediction_mode", "direct")},
        {"field": "transformer_dim", "value": metadata.get("transformer_dim")},
        {"field": "transformer_layers", "value": metadata.get("transformer_layers")},
        {"field": "transformer_heads", "value": metadata.get("transformer_heads")},
        {"field": "transformer_mlp_dim", "value": metadata.get("transformer_mlp_dim")},
        {"field": "max_tokens", "value": metadata.get("max_tokens")},
        {"field": "token_loss_weight", "value": metadata.get("token_loss_weight")},
        {"field": "train_transitions", "value": split_counts["train"]},
        {"field": "val_transitions", "value": split_counts["val"]},
        {"field": "test_transitions", "value": split_counts["test"]},
        {
            "field": "test_predictions_manifest",
            "value": prediction_outputs.get("test_predictions_manifest"),
        },
        {
            "field": "test_predictions_dir",
            "value": prediction_outputs.get("test_predictions_dir"),
        },
    ]
    summary_rows.extend(
        {"field": f"arg/{key}", "value": value}
        for key, value in metadata["args"].items()
    )
    summary = pd.DataFrame(summary_rows)
    test = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in sorted(test_metrics.items())]
    )
    clinical_columns = pd.DataFrame({"clinical_column": metadata["clinical_columns"]})
    feature_stats = pd.DataFrame(
        [
            {"clinical_column": key, **value}
            for key, value in metadata["feature_stats"].items()
        ]
    )
    predictions_path = prediction_outputs.get("test_predictions_manifest")
    predictions = (
        pd.read_csv(predictions_path)
        if predictions_path and Path(predictions_path).exists()
        else pd.DataFrame()
    )

    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    test.to_csv(output_dir / "test_metrics.csv", index=False)
    clinical_columns.to_csv(output_dir / "clinical_columns.csv", index=False)
    feature_stats.to_csv(output_dir / "clinical_feature_stats.csv", index=False)

    excel_path = output_dir / "experiment_report.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            test.to_excel(writer, sheet_name="test_metrics", index=False)
            if not predictions.empty:
                predictions.to_excel(writer, sheet_name="test_predictions", index=False)
            clinical_columns.to_excel(writer, sheet_name="clinical_columns", index=False)
            feature_stats.to_excel(writer, sheet_name="clinical_feature_stats", index=False)
    except ImportError:
        print("openpyxl is not installed; wrote CSV reports only.")


def _batch_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _prediction_filename(
    index: int,
    patient_id: str,
    source_series_uid: str,
    target_series_uid: str,
) -> str:
    digest = hashlib.sha1(
        f"{patient_id}|{source_series_uid}|{target_series_uid}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{index:06d}_{patient_id}_{digest}.npy"


def export_test_predictions(
    checkpoint_path: str,
    datamodule: LatentTransitionDataModule,
    output_dir: Path,
) -> dict[str, str] | None:
    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        return None

    predictions_dir = output_dir / "test_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "test_predictions.csv"

    model = LatentPredictorModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cpu()

    rows = []
    sample_index = 0
    test_df = datamodule.test_df.reset_index(drop=True) if datamodule.test_df is not None else None
    with torch.inference_mode():
        for batch in test_loader:
            source = batch["source"].float()
            target = batch["target"].float()
            clinical = batch["clinical"].float()
            delta = batch["delta"].float()
            pred = model(source, clinical, delta)
            pred_pooled = model._pool_if_tokens(pred)
            source_pooled = model._pool_if_tokens(source)
            target_pooled = model._pool_if_tokens(target)

            sample_l1 = torch.mean(torch.abs(pred_pooled - target_pooled), dim=1)
            sample_cosine = 1.0 - F.cosine_similarity(pred_pooled, target_pooled, dim=-1)
            sample_persistence_l1 = torch.mean(torch.abs(source_pooled - target_pooled), dim=1)
            sample_persistence_cosine = 1.0 - F.cosine_similarity(source_pooled, target_pooled, dim=-1)

            patient_ids = _batch_strings(batch["patient_id"])
            source_series_uids = _batch_strings(batch["source_series_uid"])
            target_series_uids = _batch_strings(batch["target_series_uid"])
            source_embedding_paths = _batch_strings(batch["source_embedding_path"])
            target_embedding_paths = _batch_strings(batch["target_embedding_path"])
            source_timepoints = _batch_strings(batch["source_timepoint"])
            target_timepoints = _batch_strings(batch["target_timepoint"])

            pred_np = pred_pooled.detach().cpu().numpy().astype(np.float32)
            for idx in range(pred_np.shape[0]):
                pred_path = predictions_dir / _prediction_filename(
                    sample_index,
                    patient_ids[idx],
                    source_series_uids[idx],
                    target_series_uids[idx],
                )
                np.save(pred_path, pred_np[idx])
                row = {
                    "patient_id": patient_ids[idx],
                    "source_timepoint": source_timepoints[idx],
                    "target_timepoint": target_timepoints[idx],
                    "source_series_uid": source_series_uids[idx],
                    "target_series_uid": target_series_uids[idx],
                    "source_embedding_path": source_embedding_paths[idx],
                    "target_embedding_path": target_embedding_paths[idx],
                    "pred_embedding_path": str(pred_path),
                    "test_l1": float(sample_l1[idx].item()),
                    "test_cosine": float(sample_cosine[idx].item()),
                    "persistence_l1": float(sample_persistence_l1[idx].item()),
                    "persistence_cosine": float(sample_persistence_cosine[idx].item()),
                    "relative_l1_improvement": float(
                        ((sample_persistence_l1[idx] - sample_l1[idx]) / sample_persistence_l1[idx].clamp_min(1e-12)).item()
                    ),
                    "relative_cosine_improvement": float(
                        ((sample_persistence_cosine[idx] - sample_cosine[idx]) / sample_persistence_cosine[idx].clamp_min(1e-12)).item()
                    ),
                }
                if test_df is not None:
                    meta = test_df.iloc[sample_index].to_dict()
                    for key, value in meta.items():
                        if key not in row and not key.endswith("_embedding_path"):
                            row[key] = value
                rows.append(row)
                sample_index += 1

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return {
        "test_predictions_manifest": str(manifest_path),
        "test_predictions_dir": str(predictions_dir),
    }


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    datamodule.setup("fit")
    if datamodule.embedding_dim is None:
        raise RuntimeError("DataModule did not infer embedding_dim.")

    model = LatentPredictorModule(
        embedding_dim=datamodule.embedding_dim,
        clinical_dim=datamodule.clinical_dim,
        delta_dim=datamodule.delta_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cosine_weight=args.cosine_weight,
        token_loss_weight=args.token_loss_weight,
        prediction_mode=args.prediction_mode,
        model_arch=args.model_arch,
        transformer_dim=args.transformer_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_mlp_dim=args.transformer_mlp_dim,
        max_tokens=args.max_tokens,
    )

    has_val = datamodule.val_df is not None and len(datamodule.val_df) > 0
    monitor = "val/loss" if has_val else "train/loss"
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-{epoch:03d}",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=build_logger(args),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        default_root_dir=str(args.output_dir),
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
    test_results = None
    has_test = datamodule.test_df is not None and len(datamodule.test_df) > 0
    if args.test and has_test:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")
    test_metrics = flatten_test_results(test_results)
    prediction_outputs = None
    if args.test and has_test and checkpoint.best_model_path:
        prediction_outputs = export_test_predictions(
            checkpoint.best_model_path,
            datamodule,
            args.output_dir,
        )

    metadata = {
        "embedding_dim": datamodule.embedding_dim,
        "clinical_dim": datamodule.clinical_dim,
        "delta_dim": datamodule.delta_dim,
        "model_arch": args.model_arch,
        "prediction_mode": args.prediction_mode,
        "transformer_dim": args.transformer_dim,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_mlp_dim": args.transformer_mlp_dim,
        "max_tokens": args.max_tokens,
        "token_loss_weight": args.token_loss_weight,
        "clinical_columns": datamodule.clinical_columns,
        "feature_stats": datamodule.feature_stats,
        "split_counts": {
            "train": 0 if datamodule.train_df is None else len(datamodule.train_df),
            "val": 0 if datamodule.val_df is None else len(datamodule.val_df),
            "test": 0 if datamodule.test_df is None else len(datamodule.test_df),
        },
        "best_model_path": checkpoint.best_model_path,
        "best_model_score": None if checkpoint.best_model_score is None else float(checkpoint.best_model_score),
        "test_results": test_metrics,
        "prediction_outputs": prediction_outputs,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    write_experiment_report(args.output_dir, metadata, test_metrics)
    print(f"Done. best_model_path={checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
