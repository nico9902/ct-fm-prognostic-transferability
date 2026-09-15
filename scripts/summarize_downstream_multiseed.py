#!/usr/bin/env python3
"""Summarize multiple downstream evaluation seeds into mean/std tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "val_auroc",
    "val_auprc",
    "val_balanced_accuracy",
    "test_auroc",
    "test_auprc",
    "test_balanced_accuracy",
    "test_sensitivity",
    "test_specificity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="seed_*/downstream_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for path in sorted(args.input_root.glob(args.pattern)):
        seed_name = path.parent.name
        seed = seed_name.removeprefix("seed_")
        df = pd.read_csv(path)
        df.insert(0, "seed", int(seed) if seed.isdigit() else seed)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No downstream result files found under {args.input_root}")

    all_results = pd.concat(frames, ignore_index=True)
    group_cols = ["outcome", "feature_set"]
    summary = (
        all_results.groupby(group_cols, dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    source = summary[summary["feature_set"] == "source_z_t"][
        ["outcome", "test_auroc_mean", "test_auprc_mean", "test_balanced_accuracy_mean"]
    ].rename(
        columns={
            "test_auroc_mean": "source_test_auroc_mean",
            "test_auprc_mean": "source_test_auprc_mean",
            "test_balanced_accuracy_mean": "source_test_balanced_accuracy_mean",
        }
    )
    comparison = summary.merge(source, on="outcome", how="left")
    comparison["delta_test_auroc_vs_source"] = (
        comparison["test_auroc_mean"] - comparison["source_test_auroc_mean"]
    )
    comparison["delta_test_auprc_vs_source"] = (
        comparison["test_auprc_mean"] - comparison["source_test_auprc_mean"]
    )
    comparison["delta_test_balanced_accuracy_vs_source"] = (
        comparison["test_balanced_accuracy_mean"]
        - comparison["source_test_balanced_accuracy_mean"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "downstream_multiseed_all_results.csv"
    summary_path = args.output_dir / "downstream_multiseed_summary.csv"
    xlsx_path = args.output_dir / "downstream_multiseed_summary.xlsx"
    all_results.to_csv(all_path, index=False)
    comparison.to_csv(summary_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            all_results.to_excel(writer, sheet_name="all_results", index=False)
            comparison.to_excel(writer, sheet_name="summary", index=False)
    except ImportError:
        print("openpyxl is not installed; wrote CSV only.")
    print(f"Done. summary={summary_path}")


if __name__ == "__main__":
    main()
