#!/usr/bin/env python3
"""
Build a longitudinal NLST CT cohort using Imaging Data Commons (IDC).

Why IDC?
--------
The dedicated NLST NBIA backend can return HTTP 500 errors even after
successful authentication. IDC hosts the NLST imaging collection and exposes
the metadata through the idc-index Python package, allowing robust local SQL
queries and selective downloads.

Install:
    pip install --upgrade idc-index pandas

Examples
--------
Test cohort of 10 longitudinally eligible patients:
    python scripts/build_nlst_manifest.py \
        --max-patients 10 \
        --seed 42 \
        --tag test_10

Test cohort of 100 longitudinally eligible patients:
    python scripts/build_nlst_manifest.py \
        --max-patients 100 \
        --seed 42 \
        --tag test_100

Full eligible cohort:
    python scripts/build_nlst_manifest.py --tag full

Download selected DICOM series:
    python scripts/build_nlst_manifest.py \
        --max-patients 100 \
        --seed 42 \
        --tag test_100 \
        --download

Download selected DICOM series and NLST clinical tables:
    python scripts/build_nlst_manifest.py \
        --max-patients 100 \
        --seed 42 \
        --tag test_100 \
        --download \
        --download-clinical

Build a balanced 500-patient cancer/non-cancer cohort for server download:
    python scripts/build_nlst_manifest.py \
        --max-patients 500 \
        --cohort-mode cancer-balanced \
        --tag pilot_500_balanced \
        --download \
        --download-clinical
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from idc_index import IDCClient
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'idc-index'. Install it with:\n"
        "  pip install --upgrade idc-index"
    ) from e

from download_nlst_clinical import (
    DEFAULT_TABLES,
    download_nlst_clinical_tables,
    download_table,
    get_s5cmd_path,
)


COLLECTION_ID = "nlst"

BAD_DESC_PATTERNS = (
    "LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW", "SCANOGRAM",
    "DOSE", "REPORT", "SCREEN SAVE", "SECONDARY CAPTURE",
    "CORONAL", "SAGITTAL", "MPR",
)


def choose_column(df: pd.DataFrame, candidates, required=False):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(
            f"Could not find any of columns {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_nlst_ct_index(client: IDCClient) -> pd.DataFrame:
    """
    Query all primary NLST CT series from IDC.

    SELECT * is intentional for the first metadata pass because IDC's index
    schema can evolve. Downstream logic resolves likely column names
    dynamically.
    """
    print("[IDC] Querying NLST CT series metadata...")
    df = client.sql_query("""
        SELECT *
        FROM index
        WHERE collection_id = 'nlst'
          AND Modality = 'CT'
    """)
    if df.empty:
        raise RuntimeError("IDC returned no NLST CT series.")
    print(f"[IDC] Retrieved {len(df):,} CT series.")
    return df


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    patient_col = choose_column(df, ["PatientID"], required=True)
    study_col = choose_column(df, ["StudyInstanceUID"], required=True)
    series_col = choose_column(df, ["SeriesInstanceUID"], required=True)

    date_col = choose_column(
        df,
        ["StudyDate", "study_date", "SeriesDate", "series_date"],
        required=False,
    )
    desc_col = choose_column(
        df,
        ["SeriesDescription", "series_description"],
        required=False,
    )
    protocol_col = choose_column(
        df,
        ["ProtocolName", "protocol_name"],
        required=False,
    )
    count_col = choose_column(
        df,
        ["instanceCount", "ImageCount", "NumberOfImages", "image_count"],
        required=False,
    )
    size_col = choose_column(
        df,
        ["series_size_MB", "FileSize", "file_size"],
        required=False,
    )
    body_col = choose_column(
        df,
        ["BodyPartExamined", "body_part_examined"],
        required=False,
    )
    manufacturer_col = choose_column(
        df,
        ["Manufacturer", "manufacturer"],
        required=False,
    )
    model_col = choose_column(
        df,
        ["ManufacturerModelName", "manufacturer_model_name"],
        required=False,
    )

    out = pd.DataFrame({
        "patient_id": df[patient_col].astype(str),
        "study_uid": df[study_col].astype(str),
        "series_uid": df[series_col].astype(str),
    })

    out["study_date"] = df[date_col] if date_col else None
    out["series_description"] = df[desc_col].fillna("").astype(str) if desc_col else ""
    out["protocol_name"] = df[protocol_col].fillna("").astype(str) if protocol_col else ""
    out["image_count"] = df[count_col] if count_col else None
    out["series_size_mb"] = df[size_col] if size_col else None
    out["body_part"] = df[body_col].fillna("").astype(str) if body_col else ""
    out["manufacturer"] = (
        df[manufacturer_col].fillna("").astype(str) if manufacturer_col else ""
    )
    out["manufacturer_model"] = (
        df[model_col].fillna("").astype(str) if model_col else ""
    )

    out = out.drop_duplicates("series_uid")
    return out


def bad_description(desc: str) -> bool:
    u = (desc or "").upper()
    return any(pattern in u for pattern in BAD_DESC_PATTERNS)


def score_series(row: pd.Series, min_images: int) -> float:
    desc = str(row.get("series_description", "") or "")
    protocol = str(row.get("protocol_name", "") or "")
    combined = f"{desc} {protocol}".upper()

    count = pd.to_numeric(row.get("image_count"), errors="coerce")
    count = float(count) if pd.notna(count) else 0.0

    if bad_description(combined):
        return -1e9

    # Rejected rows must stay below the cutoff used by
    # ``choose_representative_series``.  Returning ``-1e8 + count`` allowed
    # every positive count below ``min_images`` to pass ``score > -1e8``.
    if count < min_images:
        return -1e9

    score = count

    if "AXIAL" in combined:
        score += 2000
    if "LUNG" in combined:
        score += 1000
    if "CHEST" in combined:
        score += 500

    return score


def choose_representative_series(df: pd.DataFrame, min_images: int) -> pd.DataFrame:
    """
    Keep one representative CT series per imaging study.

    This is metadata-level QC only. DICOM geometry QC must follow after
    download.
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: score_series(r, min_images), axis=1)
    df = df[df["score"] > -1e8].copy()

    df["image_count_num"] = pd.to_numeric(df["image_count"], errors="coerce").fillna(0)
    df["study_date_parsed"] = pd.to_datetime(df["study_date"], errors="coerce")

    df = (
        df.sort_values(
            ["patient_id", "study_uid", "score", "image_count_num", "series_uid"],
            ascending=[True, True, False, False, True],
        )
        .drop_duplicates(["patient_id", "study_uid"], keep="first")
        .copy()
    )
    return df


def order_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    chunks = []

    for pid, g in df.groupby("patient_id", sort=True):
        g = g.copy()

        # Prefer temporal ordering from preserved DICOM study dates.
        # If dates are missing, StudyInstanceUID is only a deterministic
        # fallback and should not be interpreted as biological order.
        g = g.sort_values(
            ["study_date_parsed", "study_uid"],
            na_position="last",
        ).reset_index(drop=True)

        g["timepoint_index"] = range(len(g))
        g["timepoint"] = [f"T{i}" for i in range(len(g))]
        chunks.append(g)

    if not chunks:
        return df.iloc[0:0].copy()

    return pd.concat(chunks, ignore_index=True)


def get_eligible_patients(ordered: pd.DataFrame, min_timepoints: int) -> list[str]:
    counts = ordered.groupby("patient_id")["study_uid"].nunique()
    return sorted(counts[counts >= min_timepoints].index.astype(str).tolist())


def load_cancer_patient_ids(
    client: IDCClient,
    clinical_output_dir: Path,
    table: str,
) -> set[str]:
    raw_dir = clinical_output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = download_table(
        get_s5cmd_path(client),
        client.idc_version,
        table,
        raw_dir,
    )
    df = pd.read_parquet(parquet_path)
    patient_col = choose_column(df, ["dicom_patient_id", "pid"], required=True)
    cancer_ids = set(df[patient_col].dropna().astype(str))
    print(
        f"[clinical] Cancer-positive patients from {table}: "
        f"{len(cancer_ids):,}"
    )
    return cancer_ids


def select_random_eligible_patients(
    ordered: pd.DataFrame,
    min_timepoints: int,
    max_patients: Optional[int],
    seed: int,
) -> list[str]:
    eligible = get_eligible_patients(ordered, min_timepoints)

    print(
        f"[cohort] Longitudinally eligible patients "
        f"(>= {min_timepoints} CT studies): {len(eligible):,}"
    )

    if max_patients is None or max_patients >= len(eligible):
        return eligible

    rng = random.Random(seed)
    chosen = sorted(rng.sample(eligible, max_patients))
    print(
        f"[cohort] Randomly selected {len(chosen):,} eligible patients "
        f"with seed={seed}."
    )
    return chosen


def select_cancer_balanced_patients(
    ordered: pd.DataFrame,
    min_timepoints: int,
    max_patients: int,
    seed: int,
    cancer_patient_ids: set[str],
) -> list[str]:
    eligible = get_eligible_patients(ordered, min_timepoints)
    eligible_set = set(eligible)
    positives = sorted(eligible_set & cancer_patient_ids)
    negatives = sorted(eligible_set - cancer_patient_ids)

    print(
        f"[cohort] Longitudinally eligible patients "
        f"(>= {min_timepoints} CT studies): {len(eligible):,}"
    )
    print(
        f"[cohort] Eligible cancer-balanced pools: "
        f"positive={len(positives):,} negative={len(negatives):,}"
    )

    n_positive = max_patients // 2
    n_negative = max_patients - n_positive
    if len(positives) < n_positive or len(negatives) < n_negative:
        raise RuntimeError(
            "Not enough patients for requested balanced cohort: "
            f"need positive={n_positive}, negative={n_negative}; "
            f"available positive={len(positives)}, negative={len(negatives)}."
        )

    rng = random.Random(seed)
    chosen_positive = rng.sample(positives, n_positive)
    chosen_negative = rng.sample(negatives, n_negative)
    chosen = sorted(chosen_positive + chosen_negative)
    print(
        f"[cohort] Selected cancer-balanced patients with seed={seed}: "
        f"positive={len(chosen_positive):,} negative={len(chosen_negative):,}"
    )
    return chosen


def build_subjects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for pid, g in df.groupby("patient_id", sort=True):
        g = g.sort_values("timepoint_index")
        row = {
            "patient_id": pid,
            "n_timepoints": g["study_uid"].nunique(),
        }
        if "cancer_label" in g.columns:
            row["cancer_label"] = int(g["cancer_label"].iloc[0])

        for _, r in g.iterrows():
            tp = r["timepoint"]
            row[f"{tp}_study_uid"] = r["study_uid"]
            row[f"{tp}_series_uid"] = r["series_uid"]
            row[f"{tp}_date"] = r["study_date"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_pairs(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []

    for pid, g in df.groupby("patient_id", sort=True):
        g = g.sort_values("timepoint_index").reset_index(drop=True)

        if mode == "consecutive":
            pairs = [(i, i + 1) for i in range(len(g) - 1)]
        else:
            pairs = [
                (i, j)
                for i in range(len(g))
                for j in range(i + 1, len(g))
            ]

        for i, j in pairs:
            a = g.iloc[i]
            b = g.iloc[j]

            da = pd.to_datetime(a["study_date"], errors="coerce")
            db = pd.to_datetime(b["study_date"], errors="coerce")
            delta = int((db - da).days) if pd.notna(da) and pd.notna(db) else None

            rows.append({
                "patient_id": pid,
                "cancer_label": int(a["cancer_label"]) if "cancer_label" in g.columns else None,
                "source_timepoint": a["timepoint"],
                "target_timepoint": b["timepoint"],
                "source_study_uid": a["study_uid"],
                "target_study_uid": b["study_uid"],
                "source_series_uid": a["series_uid"],
                "target_series_uid": b["series_uid"],
                "source_date": a["study_date"],
                "target_date": b["study_date"],
                "delta_days": delta,
                "source_image_count": a["image_count"],
                "target_image_count": b["image_count"],
                "source_series_size_mb": a["series_size_mb"],
                "target_series_size_mb": b["series_size_mb"],
            })

    return pd.DataFrame(rows)


def estimate_download_size(df: pd.DataFrame):
    sizes = pd.to_numeric(df["series_size_mb"], errors="coerce")
    if sizes.notna().any():
        total_mb = sizes.sum()
        print(
            f"[download] Estimated selected-series size: "
            f"{total_mb/1024:.2f} GiB ({total_mb/1024/1024:.3f} TiB)"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-timepoints", type=int, default=2)
    parser.add_argument("--min-images", type=int, default=50)
    parser.add_argument(
        "--cohort-mode",
        choices=["random", "cancer-balanced"],
        default="random",
        help="Patient sampling strategy after longitudinal imaging eligibility.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["consecutive", "all"],
        default="consecutive",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", default="/ssd2/domenico/datasets/nlst/manifests")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the selected DICOM series after building the manifest.",
    )
    parser.add_argument(
        "--download-dir",
        default="/ssd2/domenico/datasets/nlst/raw/nlst/dicom",
    )
    parser.add_argument(
        "--dry-run-download",
        action="store_true",
        help="Ask IDC to estimate/check the selected download without downloading.",
    )
    parser.add_argument(
        "--download-clinical",
        action="store_true",
        help="Download NLST clinical tables and subset them to the selected cohort.",
    )
    parser.add_argument(
        "--clinical-output-dir",
        default="/ssd2/domenico/datasets/nlst/metadata/nlst/clinical",
        help="Directory for downloaded NLST clinical tables.",
    )
    parser.add_argument(
        "--clinical-tables",
        nargs="+",
        default=list(DEFAULT_TABLES),
        help="NLST clinical tables to download when --download-clinical is set.",
    )
    parser.add_argument(
        "--balance-table",
        default="nlst_canc",
        help="Clinical table defining cancer-positive patients for cancer-balanced cohorts.",
    )

    args = parser.parse_args()

    if args.max_patients is not None and args.max_patients <= 0:
        parser.error("--max-patients must be > 0")
    if args.cohort_mode == "cancer-balanced" and args.max_patients is None:
        parser.error("--cohort-mode cancer-balanced requires --max-patients")

    print("[IDC] Initializing IDC index...")
    client = IDCClient()

    raw = load_nlst_ct_index(client)
    norm = normalize_index(raw)

    print(f"[QC] Unique NLST CT series: {len(norm):,}")

    selected = choose_representative_series(norm, args.min_images)
    print(
        f"[QC] Representative CT studies after metadata QC: "
        f"{len(selected):,}"
    )

    ordered = order_timepoints(selected)

    cancer_patient_ids: set[str] = set()
    if args.cohort_mode == "cancer-balanced":
        cancer_patient_ids = load_cancer_patient_ids(
            client=client,
            clinical_output_dir=Path(args.clinical_output_dir),
            table=args.balance_table,
        )
        chosen_pids = select_cancer_balanced_patients(
            ordered,
            min_timepoints=args.min_timepoints,
            max_patients=args.max_patients,
            seed=args.seed,
            cancer_patient_ids=cancer_patient_ids,
        )
    else:
        chosen_pids = select_random_eligible_patients(
            ordered,
            min_timepoints=args.min_timepoints,
            max_patients=args.max_patients,
            seed=args.seed,
        )

    cohort = ordered[
        ordered["patient_id"].astype(str).isin(chosen_pids)
    ].copy()
    if args.cohort_mode == "cancer-balanced":
        cohort["cancer_label"] = (
            cohort["patient_id"].astype(str).isin(cancer_patient_ids).astype(int)
        )

    # Re-assign T0, T1, ... after final cohort selection
    cohort = order_timepoints(cohort)

    tag = args.tag or (
        f"test_{args.max_patients}"
        if args.max_patients is not None
        else "full"
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    series_path = outdir / f"nlst_{tag}_series.csv"
    subjects_path = outdir / f"nlst_{tag}_subjects.csv"
    pairs_path = outdir / f"nlst_{tag}_pairs.csv"
    uid_path = outdir / f"nlst_{tag}_series_uids.txt"

    cohort.to_csv(series_path, index=False)
    build_subjects(cohort).to_csv(subjects_path, index=False)

    pairs = build_pairs(cohort, args.pair_mode)
    pairs.to_csv(pairs_path, index=False)

    uids = sorted(cohort["series_uid"].dropna().astype(str).unique().tolist())
    uid_path.write_text("\n".join(uids) + "\n")

    print("\n=== NLST longitudinal cohort ===")
    print(f"Patients:        {cohort['patient_id'].nunique():,}")
    print(f"Selected series: {len(cohort):,}")
    print(f"Pairs:           {len(pairs):,}")
    print(f"Series CSV:      {series_path}")
    print(f"Subjects CSV:    {subjects_path}")
    print(f"Pairs CSV:       {pairs_path}")
    print(f"UID list:        {uid_path}")

    estimate_download_size(cohort)

    if args.download or args.dry_run_download:
        download_dir = Path(args.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        print(
            "[download] Passing selected SeriesInstanceUIDs directly to IDC..."
        )

        client.download_from_selection(
            seriesInstanceUID=uids,
            downloadDir=str(download_dir),
            dry_run=args.dry_run_download,
            quiet=False,
            show_progress_bar=True,
        )

        if args.dry_run_download:
            print("[download] Dry run completed; no DICOM files were downloaded.")
        else:
            print(f"[download] DICOM download completed in {download_dir}")

    if args.download_clinical:
        print("[clinical] Downloading NLST clinical tables...")
        patient_ids = set(cohort["patient_id"].dropna().astype(str))
        summary_path = download_nlst_clinical_tables(
            client=client,
            output_dir=Path(args.clinical_output_dir),
            tables=args.clinical_tables,
            patient_ids=patient_ids,
            subset_label=tag,
        )
        print(f"[clinical] Clinical tables ready: {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
