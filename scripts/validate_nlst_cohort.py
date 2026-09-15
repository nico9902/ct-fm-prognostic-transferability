#!/usr/bin/env python3
"""Validate NLST cohort balance and patient-level split integrity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lung_jepa.data.latent_transitions import split_by_patient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, default=None)
    parser.add_argument("--expected-per-class", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assert_no_overlap(left: set[str], right: set[str], name: str) -> None:
    overlap = left & right
    if overlap:
        examples = sorted(overlap)[:10]
        raise SystemExit(f"Patient overlap in {name}: {len(overlap)} examples={examples}")


def main() -> None:
    args = parse_args()
    subjects = pd.read_csv(args.subjects, dtype={"patient_id": str})
    transitions = pd.read_csv(args.transitions, dtype={"patient_id": str})

    n_patients = subjects["patient_id"].nunique()
    if args.expected_patients is not None and n_patients != args.expected_patients:
        raise SystemExit(f"Expected {args.expected_patients} patients, found {n_patients}.")

    if "cancer_label" not in subjects.columns:
        raise SystemExit("Missing cancer_label in subjects table.")

    counts = subjects.drop_duplicates("patient_id")["cancer_label"].value_counts().sort_index()
    if set(counts.index.astype(int)) != {0, 1}:
        raise SystemExit(f"Expected binary cancer_label 0/1, found counts={counts.to_dict()}.")
    if counts.loc[0] != counts.loc[1]:
        raise SystemExit(f"Cohort is not balanced: counts={counts.to_dict()}.")
    if args.expected_per_class is not None:
        if counts.loc[0] != args.expected_per_class or counts.loc[1] != args.expected_per_class:
            raise SystemExit(
                f"Expected {args.expected_per_class} patients per class, "
                f"found counts={counts.to_dict()}."
            )

    train, val, test = split_by_patient(
        transitions,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_ids = set(train["patient_id"].astype(str))
    val_ids = set(val["patient_id"].astype(str))
    test_ids = set(test["patient_id"].astype(str))
    assert_no_overlap(train_ids, val_ids, "train/val")
    assert_no_overlap(train_ids, test_ids, "train/test")
    assert_no_overlap(val_ids, test_ids, "val/test")

    print("Cohort validation passed.")
    print(f"patients={n_patients}")
    print(f"balance={counts.to_dict()}")
    print(
        "split_patients="
        f"train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}"
    )
    print(
        "split_transitions="
        f"train:{len(train)} val:{len(val)} test:{len(test)}"
    )


if __name__ == "__main__":
    main()
