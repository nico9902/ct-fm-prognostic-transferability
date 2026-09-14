#!/usr/bin/env python3
"""Audit expected NLST series against generated NIfTI/Rad-JEPA manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-manifest", type=Path, required=True)
    parser.add_argument("--nifti-manifest", type=Path, required=True)
    parser.add_argument("--radjepa-manifest", type=Path, default=None)
    parser.add_argument("--patient-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = pd.read_csv(args.series_manifest, dtype=str)
    nifti = pd.read_csv(args.nifti_manifest, dtype=str)
    if args.patient_id is not None:
        series = series[series["patient_id"].astype(str) == str(args.patient_id)]
        nifti = nifti[nifti["patient_id"].astype(str) == str(args.patient_id)]

    merged = series[["patient_id", "timepoint", "study_uid", "series_uid"]].merge(
        nifti,
        on=["patient_id", "timepoint", "study_uid", "series_uid"],
        how="left",
        suffixes=("", "_nifti"),
    )
    merged["nifti_exists"] = merged["nifti_path"].fillna("").map(
        lambda p: Path(p).exists() if p else False
    )
    missing = merged[
        (merged["status"].fillna("") == "error")
        | (~merged["nifti_exists"])
        | merged["status"].isna()
    ]

    print(f"expected_series={len(series)} nifti_rows={len(nifti)}")
    print(f"missing_or_error_nifti={len(missing)}")
    if len(missing):
        print(missing[["patient_id", "timepoint", "series_uid", "status", "error"]].to_string(index=False))

    if args.radjepa_manifest is not None and args.radjepa_manifest.exists():
        radjepa = pd.read_csv(args.radjepa_manifest, dtype=str)
        if args.patient_id is not None:
            radjepa = radjepa[radjepa["patient_id"].astype(str) == str(args.patient_id)]
        bad = radjepa[radjepa["radjepa_status"].fillna("") != "prepared"]
        print(f"radjepa_rows={len(radjepa)} radjepa_not_prepared={len(bad)}")
        if len(bad):
            print(bad[["patient_id", "timepoint", "series_uid", "radjepa_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
