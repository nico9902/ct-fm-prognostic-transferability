#!/usr/bin/env python3
"""
Build a transition table for latent world-model training.

The output CSV links each longitudinal pair to source/target Rad-JEPA embedding
files and patient-level clinical features. It is intentionally table-first so
training scripts can consume it without rejoining metadata each run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/manifests/nlst_test_10_pairs.csv"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/embeddings/radjepa3d_manifest.csv"),
    )
    parser.add_argument(
        "--clinical-prsn",
        type=Path,
        default=Path("data/metadata/nlst/clinical/cohort_subsets/test_10/nlst_prsn.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/nlst/radjepa_transitions.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs, dtype=str)
    embeddings = pd.read_csv(args.embeddings, dtype=str)

    lookup = embeddings[["series_uid", "embedding_path"]].drop_duplicates("series_uid")
    source = lookup.rename(columns={
        "series_uid": "source_series_uid",
        "embedding_path": "source_embedding_path",
    })
    target = lookup.rename(columns={
        "series_uid": "target_series_uid",
        "embedding_path": "target_embedding_path",
    })

    out = pairs.merge(source, on="source_series_uid", how="left")
    out = out.merge(target, on="target_series_uid", how="left")

    if args.clinical_prsn.exists():
        clinical = pd.read_csv(args.clinical_prsn, dtype=str)
        clinical = clinical.add_prefix("clinical_")
        clinical = clinical.rename(columns={"clinical_dicom_patient_id": "patient_id"})
        out = out.merge(clinical, on="patient_id", how="left")
    else:
        print(f"[warn] clinical table not found: {args.clinical_prsn}")

    missing_source = out["source_embedding_path"].isna().sum()
    missing_target = out["target_embedding_path"].isna().sum()
    if missing_source or missing_target:
        print(f"[warn] missing embeddings: source={missing_source} target={missing_target}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Done. transitions={len(out)} output={args.output}")


if __name__ == "__main__":
    main()
