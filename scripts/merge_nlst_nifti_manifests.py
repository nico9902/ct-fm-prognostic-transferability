#!/usr/bin/env python3
"""Merge a clean NIfTI manifest with successfully repaired/replaced series."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY = ["patient_id", "timepoint", "study_uid"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ready-manifest", type=Path, required=True)
    parser.add_argument(
        "--additional-manifest", type=Path, action="append", default=[], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(args.base_ready_manifest, dtype=str)]
    for path in args.additional_manifest:
        frame = pd.read_csv(path, dtype=str)
        frame = frame[frame["status"].fillna("").isin(["converted", "exists"])]
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    duplicates = merged.duplicated(KEY, keep=False)
    if duplicates.any():
        rows = merged.loc[duplicates, KEY + ["series_uid"]]
        raise SystemExit(f"Duplicate study/timepoint rows after merge:\n{rows.to_string(index=False)}")
    merged = merged.sort_values(KEY).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(
        f"Merged NIfTI rows={len(merged)} patients={merged['patient_id'].nunique()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
