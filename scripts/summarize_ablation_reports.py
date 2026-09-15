#!/usr/bin/env python3
"""Collect latent predictor ablation reports into one comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing individual experiment output folders.",
    )
    parser.add_argument(
        "--pattern",
        default="latent_predictor_pilot_500_balanced_*_gpu1",
        help="Glob pattern for individual run directories under --run-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation_pilot_500_balanced_gpu1"),
    )
    return parser.parse_args()


def read_key_value_csv(path: Path, key_col: str, value_col: str) -> dict[str, object]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if key_col not in df.columns or value_col not in df.columns:
        return {}
    return dict(zip(df[key_col], df[value_col], strict=False))


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(args.run_root.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        summary = read_key_value_csv(run_dir / "experiment_summary.csv", "field", "value")
        metrics = read_key_value_csv(run_dir / "test_metrics.csv", "metric", "value")
        row = {
            "experiment": run_dir.name,
            "output_dir": str(run_dir),
            "use_clinical": summary.get("arg/use_clinical"),
            "use_delta": summary.get("arg/use_delta"),
            "prediction_mode": summary.get("prediction_mode") or summary.get("arg/prediction_mode"),
            "clinical_dim": summary.get("clinical_dim"),
            "delta_dim": summary.get("delta_dim"),
            "best_model_score": summary.get("best_model_score"),
            "test_loss": metrics.get("test/loss"),
            "test_l1": metrics.get("test/l1"),
            "test_cosine": metrics.get("test/cosine"),
            "persistence_loss": metrics.get("test/persistence_loss"),
            "persistence_l1": metrics.get("test/persistence_l1"),
            "persistence_cosine": metrics.get("test/persistence_cosine"),
            "relative_loss_improvement": metrics.get("test/relative_loss_improvement"),
            "relative_l1_improvement": metrics.get("test/relative_l1_improvement"),
            "relative_cosine_improvement": metrics.get("test/relative_cosine_improvement"),
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"No experiment reports found under {args.run_root} with pattern {args.pattern}")

    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_summary.csv"
    xlsx_path = args.output_dir / "ablation_summary.xlsx"
    out.to_csv(csv_path, index=False)
    try:
        out.to_excel(xlsx_path, index=False)
    except ImportError:
        print("openpyxl is not installed; wrote CSV summary only.")
    print(f"Done. summary={csv_path}")


if __name__ == "__main__":
    main()
