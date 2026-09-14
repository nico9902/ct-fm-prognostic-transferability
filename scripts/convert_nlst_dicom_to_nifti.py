#!/usr/bin/env python3
"""
Convert NLST DICOM CT series to compressed NIfTI volumes.

Example:
    python scripts/convert_nlst_dicom_to_nifti.py \
        --manifest data/manifests/nlst_test_10_series.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError


LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def find_series_dir(dicom_root: Path, patient_id: str, study_uid: str, series_uid: str) -> Path:
    expected = dicom_root / "nlst" / patient_id / study_uid / f"CT_{series_uid}"
    if expected.is_dir():
        return expected

    patient_root = dicom_root / "nlst" / patient_id
    matches = list(patient_root.glob(f"*/CT_{series_uid}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No DICOM directory found for series {series_uid}")
    raise RuntimeError(f"Multiple DICOM directories found for series {series_uid}: {matches}")


def read_dicom_slices(series_dir: Path) -> list[pydicom.dataset.FileDataset]:
    slices = []
    for path in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(path)
        except InvalidDicomError:
            continue
        if hasattr(ds, "PixelData"):
            slices.append(ds)

    if not slices:
        raise RuntimeError(f"No readable DICOM slices with pixel data in {series_dir}")

    return slices


def slice_position(ds: pydicom.dataset.FileDataset, normal: np.ndarray) -> float:
    if hasattr(ds, "ImagePositionPatient"):
        return float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
    return float(getattr(ds, "InstanceNumber", 0))


def sort_slices(
    slices: list[pydicom.dataset.FileDataset],
) -> tuple[list[pydicom.dataset.FileDataset], np.ndarray, np.ndarray, np.ndarray]:
    first = slices[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    normal = np.cross(row_cos, col_cos)

    slices = sorted(slices, key=lambda ds: slice_position(ds, normal))
    positions = np.asarray([slice_position(ds, normal) for ds in slices], dtype=float)

    return slices, row_cos, col_cos, positions


def infer_slice_spacing(
    slices: list[pydicom.dataset.FileDataset],
    positions: np.ndarray,
) -> float:
    diffs = np.diff(positions)
    diffs = np.abs(diffs[np.abs(diffs) > 1e-3])
    if len(diffs):
        return float(np.median(diffs))

    first = slices[0]
    spacing_between = getattr(first, "SpacingBetweenSlices", None)
    if spacing_between is not None:
        return float(spacing_between)

    slice_thickness = getattr(first, "SliceThickness", None)
    if slice_thickness is not None:
        return float(slice_thickness)

    return 1.0


def build_affine(
    slices: list[pydicom.dataset.FileDataset],
    row_cos: np.ndarray,
    col_cos: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    first = slices[0]
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    normal = np.cross(row_cos, col_cos)
    if len(positions) > 1 and positions[-1] < positions[0]:
        normal *= -1.0
    slice_spacing = infer_slice_spacing(slices, positions)

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, 0] = col_cos * row_spacing
    affine_lps[:3, 1] = row_cos * col_spacing
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = np.asarray(first.ImagePositionPatient, dtype=float)

    return LPS_TO_RAS @ affine_lps


def stack_hu(slices: list[pydicom.dataset.FileDataset]) -> np.ndarray:
    arrays = []
    use_float = False
    for ds in slices:
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        if slope != 1.0 or not intercept.is_integer():
            use_float = True
        pixels = pixels * slope + intercept
        arrays.append(pixels)

    volume = np.stack(arrays, axis=-1)
    if use_float:
        return volume.astype(np.float32)
    return np.rint(volume).astype(np.int16)


def convert_series(
    row: pd.Series,
    dicom_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, str]:
    patient_id = str(row.patient_id)
    study_uid = str(row.study_uid)
    series_uid = str(row.series_uid)
    timepoint = str(row.timepoint)

    series_dir = find_series_dir(dicom_root, patient_id, study_uid, series_uid)
    out_dir = output_root / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}_{timepoint}_{series_uid}.nii.gz"

    if out_path.exists() and not overwrite:
        image = nib.load(out_path)
        return {
            "patient_id": patient_id,
            "timepoint": timepoint,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "dicom_dir": str(series_dir),
            "nifti_path": str(out_path),
            "shape": "x".join(map(str, image.shape)),
            "dtype": str(image.get_data_dtype()),
            "status": "exists",
            "error": "",
        }

    slices = read_dicom_slices(series_dir)
    slices, row_cos, col_cos, positions = sort_slices(slices)
    volume = stack_hu(slices)
    affine = build_affine(slices, row_cos, col_cos, positions)

    image = nib.Nifti1Image(volume, affine)
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = f"NLST CT {patient_id} {timepoint}"[:80].encode("ascii")
    nib.save(image, out_path)

    return {
        "patient_id": patient_id,
        "timepoint": timepoint,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "dicom_dir": str(series_dir),
        "nifti_path": str(out_path),
        "shape": "x".join(map(str, volume.shape)),
        "dtype": str(volume.dtype),
        "status": "converted",
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/nlst_test_10_series.csv"),
        help="Series manifest CSV produced by build_nlst_manifest.py.",
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=Path("data/raw/nlst/dicom"),
        help="Root containing the NLST DICOM download.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/nlst/nii"),
        help="Directory where .nii.gz files will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/nlst/nii_manifest.csv"),
        help="CSV describing generated NIfTI files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing NIfTI files.")
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Convert only rows for this patient_id.",
    )
    parser.add_argument(
        "--timepoint",
        default=None,
        help="Convert only rows for this timepoint, for example T1.",
    )
    parser.add_argument(
        "--series-uid",
        default=None,
        help="Convert only this SeriesInstanceUID.",
    )
    parser.add_argument(
        "--on-error",
        choices=["fail", "skip"],
        default="fail",
        help="Fail on the first conversion error, or record the error and continue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.patient_id is not None:
        manifest = manifest[manifest["patient_id"].astype(str) == str(args.patient_id)]
    if args.timepoint is not None:
        manifest = manifest[manifest["timepoint"].astype(str) == str(args.timepoint)]
    if args.series_uid is not None:
        manifest = manifest[manifest["series_uid"].astype(str) == str(args.series_uid)]
    if manifest.empty:
        raise SystemExit("No manifest rows match the requested filters.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(manifest)
    for display_index, (_, row) in enumerate(manifest.iterrows(), start=1):
        try:
            result = convert_series(row, args.dicom_root, args.output_root, args.overwrite)
        except Exception as e:
            if args.on_error == "fail":
                raise
            result = {
                "patient_id": str(row.get("patient_id", "")),
                "timepoint": str(row.get("timepoint", "")),
                "study_uid": str(row.get("study_uid", "")),
                "series_uid": str(row.get("series_uid", "")),
                "dicom_dir": "",
                "nifti_path": "",
                "shape": "",
                "dtype": "",
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(result)
        target = result["nifti_path"] or result["series_uid"]
        print(
            f"[{display_index:03d}/{total:03d}] "
            f"{result['status']}: {result['patient_id']} {result['timepoint']} -> "
            f"{target}"
        )
        if result["error"]:
            print(f"  error: {result['error']}")

    fields = [
        "patient_id",
        "timepoint",
        "study_uid",
        "series_uid",
        "dicom_dir",
        "nifti_path",
        "shape",
        "dtype",
        "status",
        "error",
    ]
    with args.output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    converted = sum(r["status"] == "converted" for r in results)
    existing = sum(r["status"] == "exists" for r in results)
    errors = sum(r["status"] == "error" for r in results)
    print(
        f"Done. converted={converted} exists={existing} "
        f"errors={errors} manifest={args.output_manifest}"
    )


if __name__ == "__main__":
    main()
