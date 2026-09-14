#!/usr/bin/env python3
"""
Download NLST clinical tables from IDC and optionally subset them to a cohort.

The Imaging Data Commons publishes NLST clinical parquet tables in a public S3
bucket. This script downloads the NLST-specific tables only, then writes CSV
subsets for the patients present in a local manifest.

Example:
    python scripts/download_nlst_clinical.py \
        --manifest data/manifests/nlst_test_10_subjects.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

try:
    from idc_index import IDCClient
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'idc-index'. Install it with:\n"
        "  pip install --upgrade idc-index"
    ) from e


DEFAULT_TABLES = (
    "nlst_prsn",
    "nlst_canc",
    "nlst_screen",
    "nlst_ctab",
    "nlst_ctabc",
)


def get_s5cmd_path(client: IDCClient) -> str:
    if client.s5cmdPath:
        return str(client.s5cmdPath)

    s5cmd_path = shutil.which("s5cmd")
    if s5cmd_path:
        return s5cmd_path

    raise RuntimeError("s5cmd executable not found. It is normally installed with idc-index.")


def download_table(s5cmd_path: str, idc_version: str, table: str, output_dir: Path) -> Path:
    table_dir = output_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(table_dir.glob("*.parquet"))
    if len(existing) == 1:
        return existing[0]

    source = f"s3://idc-open-metadata/bigquery_export/idc_{idc_version}_clinical/{table}/*"

    subprocess.run(
        [s5cmd_path, "--no-sign-request", "cp", source, str(table_dir)],
        check=True,
    )

    parquet_files = sorted(table_dir.glob("*.parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(f"Expected one parquet file for {table}, found {len(parquet_files)}")
    return parquet_files[0]


def load_patient_ids(manifest: Path | None) -> set[str] | None:
    if manifest is None:
        return None

    df = pd.read_csv(manifest, dtype=str)
    if "patient_id" not in df.columns:
        raise KeyError(f"Manifest {manifest} does not contain a patient_id column.")
    return set(df["patient_id"].dropna().astype(str))


def subset_table(df: pd.DataFrame, patient_ids: set[str] | None) -> pd.DataFrame:
    if patient_ids is None:
        return df

    key = "dicom_patient_id" if "dicom_patient_id" in df.columns else "pid"
    if key not in df.columns:
        return df.iloc[0:0].copy()

    return df[df[key].dropna().astype(str).isin(patient_ids)].copy()


def download_nlst_clinical_tables(
    client: IDCClient,
    output_dir: Path,
    tables: list[str],
    patient_ids: set[str] | None = None,
    subset_label: str = "cohort",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    subset_dir = output_dir / "cohort_subsets" / subset_label
    raw_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    s5cmd_path = get_s5cmd_path(client)

    summary_rows = []
    for table in tables:
        parquet_path = download_table(s5cmd_path, client.idc_version, table, raw_dir)
        df = pd.read_parquet(parquet_path)
        subset = subset_table(df, patient_ids)
        subset_path = subset_dir / f"{table}.csv"
        subset.to_csv(subset_path, index=False)

        summary_rows.append({
            "table": table,
            "raw_path": str(parquet_path),
            "raw_rows": len(df),
            "raw_columns": len(df.columns),
            "subset_path": str(subset_path),
            "subset_rows": len(subset),
            "subset_columns": len(subset.columns),
        })
        print(
            f"{table}: raw={len(df)}x{len(df.columns)} "
            f"subset={len(subset)}x{len(subset.columns)}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"clinical_tables_summary_{subset_label}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/nlst_test_10_subjects.csv"),
        help="Optional cohort manifest containing patient_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metadata/nlst/clinical"),
        help="Directory for downloaded tables and CSV subsets.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(DEFAULT_TABLES),
        help="NLST clinical table names to download.",
    )
    parser.add_argument(
        "--subset-label",
        default=None,
        help="Name for the cohort subset folder and summary suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = IDCClient()
    patient_ids = load_patient_ids(args.manifest)
    subset_label = args.subset_label or args.manifest.stem
    download_nlst_clinical_tables(
        client=client,
        output_dir=args.output_dir,
        tables=args.tables,
        patient_ids=patient_ids,
        subset_label=subset_label,
    )


if __name__ == "__main__":
    main()
