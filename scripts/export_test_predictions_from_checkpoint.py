#!/usr/bin/env python3
"""Export held-out test predictions from an existing latent predictor checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lung_jepa.data.latent_transitions import LatentTransitionDataModule
from lung_jepa.training.train_latent_predictor import (
    export_test_predictions,
    write_experiment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--transitions",
        type=Path,
        default=Path("/ssd2/domenico/datasets/nlst/processed/nlst/pilot_500_balanced/radjepa_transitions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-clinical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prediction-mode",
        choices=["direct", "residual"],
        default=None,
        help="Optional metadata only; checkpoint hparams define model behavior.",
    )
    return parser.parse_args()


def read_metadata(output_dir: Path) -> dict:
    path = output_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
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
    datamodule.setup("test")
    prediction_outputs = export_test_predictions(
        str(args.checkpoint),
        datamodule,
        args.output_dir,
    )

    metadata = read_metadata(args.output_dir)
    if metadata:
        metadata["prediction_outputs"] = prediction_outputs
        metadata.setdefault("delta_dim", datamodule.delta_dim)
        metadata.setdefault("clinical_dim", datamodule.clinical_dim)
        metadata.setdefault("clinical_columns", datamodule.clinical_columns)
        metadata.setdefault("feature_stats", datamodule.feature_stats)
        metadata.setdefault(
            "split_counts",
            {
                "train": 0 if datamodule.train_df is None else len(datamodule.train_df),
                "val": 0 if datamodule.val_df is None else len(datamodule.val_df),
                "test": 0 if datamodule.test_df is None else len(datamodule.test_df),
            },
        )
        test_metrics = metadata.get("test_results", {})
        with (args.output_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        write_experiment_report(args.output_dir, metadata, test_metrics)

    print(f"Done. predictions={prediction_outputs['test_predictions_manifest']}")


if __name__ == "__main__":
    main()
