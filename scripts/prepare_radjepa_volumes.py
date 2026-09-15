#!/usr/bin/env python3
"""
Prepare NLST NIfTI volumes for Rad-JEPA-3D.

Rad-JEPA-3D expects volumes shaped (1, 32, 256, 256), float32, normalized to
[0, 1]. This script reads the NIfTI manifest produced by
convert_nlst_dicom_to_nifti.py and writes one .npy volume per CT series.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


TARGET_SHAPE = (32, 256, 256)


def normalize_volume(volume: np.ndarray, mode: str, window: tuple[float, float]) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    if mode == "window":
        low, high = window
        volume = np.clip(volume, low, high)
        return (volume - low) / max(high - low, 1e-6)

    low = float(np.nanmin(volume))
    high = float(np.nanmax(volume))
    return (volume - low) / max(high - low, 1e-6)


def resize_dhw(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    factors = [target / current for target, current in zip(target_shape, volume.shape)]
    return zoom(volume, zoom=factors, order=1).astype(np.float32, copy=False)


def prepare_one(
    nifti_path: Path,
    output_path: Path,
    normalization: str,
    window: tuple[float, float],
    overwrite: bool,
    canonicalize: bool,
) -> tuple[str, str, dict[str, str]]:
    image = nib.load(str(nifti_path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape={image.shape}")
    source_orientation = "".join(nib.aff2axcodes(image.affine))
    if canonicalize:
        image = nib.as_closest_canonical(image)
    prepared_orientation = "".join(nib.aff2axcodes(image.affine))
    source_spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    source_shape = tuple(int(v) for v in image.shape)
    metadata = {
        "radjepa_source_orientation": source_orientation,
        "radjepa_prepared_orientation": prepared_orientation,
        "radjepa_source_spacing_mm": "x".join(f"{v:.6g}" for v in source_spacing),
        "radjepa_effective_spacing_mm": "x".join(
            f"{source_shape[axis] * source_spacing[axis] / target:.6g}"
            for axis, target in zip((2, 0, 1), TARGET_SHAPE)
        ),
        "radjepa_canonicalized": str(bool(canonicalize)).lower(),
        "radjepa_preprocessing_version": "nlst_radjepa_v2",
    }

    if output_path.exists() and not overwrite:
        arr = np.load(output_path, mmap_mode="r")
        # The .npy format does not retain orientation or preprocessing
        # provenance, so an older file cannot be certified as v2 here.
        metadata["radjepa_canonicalized"] = "unknown"
        metadata["radjepa_preprocessing_version"] = "existing_unverified"
        return "exists", "x".join(map(str, arr.shape)), metadata

    volume = np.asarray(image.dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=-1000.0, posinf=window[1], neginf=window[0])

    # NIfTI is loaded as (X, Y, Z); Rad-JEPA uses (C, D, H, W).
    volume = np.moveaxis(volume, -1, 0)
    volume = normalize_volume(volume, normalization, window)
    volume = resize_dhw(volume, TARGET_SHAPE)
    volume = volume[None, ...].astype(np.float32, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, volume)
    return "prepared", "x".join(map(str, volume.shape)), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nifti-manifest",
        type=Path,
        default=Path("data/processed/nlst/nii_manifest.csv"),
        help="Manifest with nifti_path entries.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/nlst/radjepa_npy"),
        help="Directory for Rad-JEPA-ready .npy volumes.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/nlst/radjepa_npy_manifest.csv"),
        help="CSV describing prepared .npy volumes.",
    )
    parser.add_argument(
        "--normalization",
        choices=["window", "minmax"],
        default="window",
        help="Use lung-style HU windowing or per-volume min-max normalization.",
    )
    parser.add_argument("--window-min", type=float, default=-1000.0)
    parser.add_argument("--window-max", type=float, default=400.0)
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Keep source array orientation instead of canonical RAS orientation.",
    )
    parser.add_argument(
        "--require-qc-ready",
        action="store_true",
        help="Require every input row to have qc_status=ready.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.nifti_manifest, dtype=str)
    if args.require_qc_ready:
        if "qc_status" not in manifest.columns:
            raise SystemExit(
                "--require-qc-ready requires a manifest produced by "
                "validate_nlst_nifti_for_radjepa.py"
            )
        bad = manifest[manifest["qc_status"].fillna("") != "ready"]
        if len(bad):
            raise SystemExit(f"Input manifest contains {len(bad)} rows that are not qc_status=ready")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, row in manifest.iterrows():
        if str(row.get("status", "")) == "error" or not str(row.get("nifti_path", "")).strip():
            result = row.to_dict()
            result.update({
                "radjepa_npy_path": "",
                "radjepa_shape": "",
                "radjepa_normalization": args.normalization,
                "radjepa_status": "skipped",
            })
            results.append(result)
            print(f"[{index + 1:03d}/{len(manifest):03d}] skipped: {row.get('series_uid', '')}")
            continue

        patient_id = str(row["patient_id"])
        timepoint = str(row["timepoint"])
        series_uid = str(row["series_uid"])
        out_dir = args.output_root / patient_id
        out_path = out_dir / f"{patient_id}_{timepoint}_{series_uid}.npy"
        status, shape, preprocessing_metadata = prepare_one(
            nifti_path=Path(row["nifti_path"]),
            output_path=out_path,
            normalization=args.normalization,
            window=(args.window_min, args.window_max),
            overwrite=args.overwrite,
            canonicalize=not args.no_canonicalize,
        )

        result = row.to_dict()
        result.update({
            "radjepa_npy_path": str(out_path),
            "radjepa_shape": shape,
            "radjepa_normalization": args.normalization,
            "radjepa_status": status,
        })
        result.update(preprocessing_metadata)
        results.append(result)
        print(f"[{index + 1:03d}/{len(manifest):03d}] {status}: {out_path}")

    fieldnames = list(dict.fromkeys(key for result in results for key in result))
    with args.output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
