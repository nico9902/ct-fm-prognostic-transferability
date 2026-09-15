#!/usr/bin/env python3
"""Validate NLST NIfTI volumes before RadJEPA-3D preprocessing.

The script never modifies source images.  It writes a row-level QC report and
a manifest containing only volumes that pass image-type, geometry, coverage,
and NIfTI/DICOM consistency checks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom


LOCALIZER_TERMS = ("LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW", "SCANOGRAM")
DICOM_QC_TAGS = [
    "Modality",
    "Rows",
    "Columns",
    "ImageType",
    "SeriesDescription",
    "ProtocolName",
    "ImageOrientationPatient",
    "ImagePositionPatient",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifti-manifest", type=Path, required=True)
    parser.add_argument("--output-ready-manifest", type=Path, required=True)
    parser.add_argument("--output-qc-report", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--min-slices", type=int, default=50)
    parser.add_argument("--min-coverage-mm", type=float, default=200.0)
    parser.add_argument(
        "--axial-normal-threshold",
        type=float,
        default=0.9,
        help="Minimum absolute superior/inferior component of the slice normal.",
    )
    parser.add_argument(
        "--max-gap-ratio",
        type=float,
        default=1.5,
        help="Maximum slice-position gap divided by the median nonzero gap.",
    )
    parser.add_argument(
        "--spacing-relative-tolerance",
        type=float,
        default=0.05,
        help="Allowed relative disagreement between NIfTI and DICOM slice spacing.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Validate only the first N rows; intended for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows; excluded rows are always printed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of series validated concurrently.",
    )
    return parser.parse_args()


def image_type_text(ds: pydicom.dataset.FileDataset) -> str:
    values = getattr(ds, "ImageType", [])
    if isinstance(values, str):
        values = [values]
    fields = [str(value) for value in values]
    fields.extend(
        [str(getattr(ds, "SeriesDescription", "")), str(getattr(ds, "ProtocolName", ""))]
    )
    return " ".join(fields).upper()


def is_localizer(ds: pydicom.dataset.FileDataset) -> bool:
    text = image_type_text(ds)
    return any(term in text for term in LOCALIZER_TERMS)


def read_headers(dicom_dir: Path) -> list[pydicom.dataset.FileDataset]:
    headers = []
    for path in sorted(dicom_dir.glob("*.dcm")):
        headers.append(
            pydicom.dcmread(path, stop_before_pixels=True, specific_tags=DICOM_QC_TAGS)
        )
    return headers


def add_reason(reasons: list[str], condition: bool, reason: str) -> None:
    if condition:
        reasons.append(reason)


def validate_row(row: pd.Series, args: argparse.Namespace) -> dict[str, object]:
    result = row.to_dict()
    reasons: list[str] = []
    warnings: list[str] = []
    nifti_path = Path(str(row.get("nifti_path", "")))
    dicom_dir = Path(str(row.get("dicom_dir", "")))

    metrics: dict[str, object] = {
        "nifti_orientation": "",
        "canonical_orientation": "",
        "nifti_spacing_x_mm": np.nan,
        "nifti_spacing_y_mm": np.nan,
        "nifti_spacing_z_mm": np.nan,
        "nifti_slices": 0,
        "dicom_files": 0,
        "diagnostic_slices": 0,
        "unique_position_slices": 0,
        "localizer_images": 0,
        "median_slice_spacing_mm": np.nan,
        "max_gap_ratio": np.nan,
        "coverage_mm": np.nan,
    }

    if not nifti_path.is_file():
        reasons.append("missing_nifti")
    else:
        try:
            image = nib.load(str(nifti_path))
            shape = image.shape
            zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
            affine = np.asarray(image.affine, dtype=float)
            add_reason(reasons, len(shape) != 3, "nifti_not_3d")
            add_reason(
                reasons,
                len(shape) == 3 and any(int(size) <= 0 for size in shape),
                "invalid_nifti_shape",
            )
            add_reason(
                reasons,
                len(zooms) != 3 or not np.isfinite(zooms).all() or (zooms <= 0).any(),
                "invalid_nifti_spacing",
            )
            add_reason(reasons, not np.isfinite(affine).all(), "invalid_nifti_affine")
            if len(shape) == 3:
                metrics["nifti_slices"] = int(shape[2])
            if len(zooms) == 3:
                metrics["nifti_spacing_x_mm"] = float(zooms[0])
                metrics["nifti_spacing_y_mm"] = float(zooms[1])
                metrics["nifti_spacing_z_mm"] = float(zooms[2])
            metrics["nifti_orientation"] = "".join(nib.aff2axcodes(affine))
            metrics["canonical_orientation"] = "".join(
                nib.aff2axcodes(nib.as_closest_canonical(image).affine)
            )
        except Exception as exc:
            reasons.append(f"unreadable_nifti:{type(exc).__name__}")

    if not dicom_dir.is_dir():
        reasons.append("missing_dicom_directory")
        headers: list[pydicom.dataset.FileDataset] = []
    else:
        try:
            headers = read_headers(dicom_dir)
        except Exception as exc:
            reasons.append(f"unreadable_dicom:{type(exc).__name__}")
            headers = []

    metrics["dicom_files"] = len(headers)
    diagnostic = [
        ds
        for ds in headers
        if str(getattr(ds, "Modality", "")).upper() == "CT" and not is_localizer(ds)
    ]
    metrics["diagnostic_slices"] = len(diagnostic)
    metrics["localizer_images"] = sum(is_localizer(ds) for ds in headers)
    add_reason(reasons, not headers, "no_dicom_headers")
    add_reason(reasons, len(diagnostic) < args.min_slices, "too_few_diagnostic_slices")

    if diagnostic:
        shapes = Counter(
            (int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0)))
            for ds in diagnostic
        )
        add_reason(reasons, len(shapes) != 1, "mixed_diagnostic_slice_shapes")

        orientations = []
        for ds in diagnostic:
            value = getattr(ds, "ImageOrientationPatient", None)
            if value is not None and len(value) == 6:
                orientations.append(np.asarray(value, dtype=float))
        add_reason(reasons, len(orientations) != len(diagnostic), "missing_slice_orientation")

        positions: list[float] = []
        if orientations:
            reference = orientations[0]
            normal = np.cross(reference[:3], reference[3:])
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 0:
                normal = normal / normal_norm
            add_reason(reasons, normal_norm <= 0, "invalid_slice_orientation")
            add_reason(
                reasons,
                normal_norm > 0 and abs(float(normal[2])) < args.axial_normal_threshold,
                "non_axial_series",
            )
            max_orientation_delta = max(
                float(np.max(np.abs(orientation - reference))) for orientation in orientations
            )
            add_reason(reasons, max_orientation_delta > 1e-4, "inconsistent_slice_orientation")

            for ds in diagnostic:
                value = getattr(ds, "ImagePositionPatient", None)
                if value is not None and len(value) == 3 and normal_norm > 0:
                    positions.append(float(np.dot(np.asarray(value, dtype=float), normal)))

        add_reason(reasons, len(positions) != len(diagnostic), "missing_slice_position")
        if len(positions) >= 2:
            ordered = np.sort(np.asarray(positions, dtype=float))
            diffs = np.diff(ordered)
            has_duplicates = bool((diffs <= 1e-3).any())
            duplicate_repair_recorded = "removed_identical_duplicate_slices=" in str(
                row.get("warning", "")
            )
            add_reason(
                reasons,
                has_duplicates and not duplicate_repair_recorded,
                "duplicate_slice_position",
            )
            if has_duplicates and duplicate_repair_recorded:
                warnings.append("identical_duplicate_slices_removed_before_nifti")
            unique_positions = [ordered[0]]
            for position in ordered[1:]:
                if position - unique_positions[-1] > 1e-3:
                    unique_positions.append(position)
            metrics["unique_position_slices"] = len(unique_positions)
            positive = diffs[diffs > 1e-3]
            if len(positive):
                median_spacing = float(np.median(positive))
                gap_ratio = float(np.max(positive) / median_spacing)
                coverage = float(ordered[-1] - ordered[0] + median_spacing)
                metrics["median_slice_spacing_mm"] = median_spacing
                metrics["max_gap_ratio"] = gap_ratio
                metrics["coverage_mm"] = coverage
                add_reason(reasons, gap_ratio > args.max_gap_ratio, "large_slice_gap")
                add_reason(reasons, coverage < args.min_coverage_mm, "insufficient_coverage")

                nifti_spacing = metrics["nifti_spacing_z_mm"]
                if np.isfinite(nifti_spacing):
                    relative_error = abs(float(nifti_spacing) - median_spacing) / median_spacing
                    add_reason(
                        reasons,
                        relative_error > args.spacing_relative_tolerance,
                        "nifti_dicom_spacing_mismatch",
                    )

    nifti_slices = int(metrics["nifti_slices"])
    diagnostic_slices = int(metrics["diagnostic_slices"])
    expected_nifti_slices = int(metrics["unique_position_slices"]) or diagnostic_slices
    add_reason(
        reasons,
        nifti_slices > 0
        and expected_nifti_slices > 0
        and nifti_slices != expected_nifti_slices,
        "nifti_diagnostic_slice_count_mismatch",
    )
    if int(metrics["localizer_images"]) > 0 and not reasons:
        warnings.append("localizer_removed_before_nifti")

    result.update(metrics)
    result["qc_status"] = "excluded" if reasons else "ready"
    result["qc_reasons"] = ";".join(sorted(set(reasons)))
    result["qc_warnings"] = ";".join(sorted(set(warnings)))
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.nifti_manifest, dtype=str)
    if args.max_rows is not None:
        manifest = manifest.head(args.max_rows)

    source_rows = [row for _, row in manifest.iterrows()]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, result in enumerate(
            executor.map(validate_row, source_rows, repeat(args), chunksize=4), start=1
        ):
            results.append(result)
            if (
                result["qc_status"] == "excluded"
                or index == 1
                or index == len(manifest)
                or (args.progress_every > 0 and index % args.progress_every == 0)
            ):
                print(
                    f"[{index:04d}/{len(manifest):04d}] {result['qc_status']}: "
                    f"{result.get('patient_id', '')} {result.get('timepoint', '')} "
                    f"{result['qc_reasons']}",
                    flush=True,
                )

    extra_fields = [
        "nifti_orientation",
        "canonical_orientation",
        "nifti_spacing_x_mm",
        "nifti_spacing_y_mm",
        "nifti_spacing_z_mm",
        "nifti_slices",
        "dicom_files",
        "diagnostic_slices",
        "unique_position_slices",
        "localizer_images",
        "median_slice_spacing_mm",
        "max_gap_ratio",
        "coverage_mm",
        "qc_status",
        "qc_reasons",
        "qc_warnings",
    ]
    fields = list(manifest.columns) + [field for field in extra_fields if field not in manifest]
    write_csv(args.output_qc_report, results, fields)
    ready = [row for row in results if row["qc_status"] == "ready"]
    write_csv(args.output_ready_manifest, ready, fields)

    reason_counts: Counter[str] = Counter()
    for row in results:
        reason_counts.update(reason for reason in str(row["qc_reasons"]).split(";") if reason)
    summary = {
        "input_rows": len(results),
        "ready_rows": len(ready),
        "excluded_rows": len(results) - len(ready),
        "thresholds": {
            "min_slices": args.min_slices,
            "min_coverage_mm": args.min_coverage_mm,
            "axial_normal_threshold": args.axial_normal_threshold,
            "max_gap_ratio": args.max_gap_ratio,
            "spacing_relative_tolerance": args.spacing_relative_tolerance,
        },
        "exclusion_reasons": dict(sorted(reason_counts.items())),
    }
    if args.output_summary is not None:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
