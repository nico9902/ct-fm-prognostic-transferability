#!/usr/bin/env python3
"""Prepare one or more NLST DICOM series for native-z sliding windows.

The output keeps the native number of axial slices.  Only the in-plane axes
are resized to 256x256, so a later 32-slice sliding window does not blur small
nodules along z.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from convert_nlst_dicom_to_nifti import (
    deduplicate_identical_positions,
    filter_localizers,
    filter_slices_by_dominant_shape,
    read_dicom_slices,
    sort_slices,
    stack_hu,
    validate_geometry,
    find_series_dir,
)


def crop_foreground(volume: np.ndarray, threshold: float = -500.0, margin: int = 8):
    """Crop (D,H,W) to the patient foreground, retaining a pixel margin.

    A robust coordinate percentile avoids one bad detector pixel or table
    artifact forcing the crop back to the full 512x512 field of view.
    """
    mask = volume > threshold
    coords = np.where(mask)
    if any(len(axis) == 0 for axis in coords):
        return volume, (0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2])
    bounds = []
    for axis_index, (axis, size) in enumerate(zip(coords, volume.shape)):
        # z is kept conservative; x/y benefit most from foreground cropping.
        qlo, qhi = (0.0, 1.0) if axis_index == 0 else (0.01, 0.99)
        lo = max(int(np.floor(np.quantile(axis, qlo))) - margin, 0)
        hi = min(int(np.ceil(np.quantile(axis, qhi))) + margin + 1, size)
        bounds.extend((lo, hi))
    d0, d1, h0, h1, w0, w1 = bounds
    return volume[d0:d1, h0:h1, w0:w1], tuple(bounds)


def resize_xy(volume: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize each axial slice in-plane, preserving native depth."""
    factors = (1.0, size / volume.shape[1], size / volume.shape[2])
    return zoom(volume, factors, order=1).astype(np.float32, copy=False)


def normalize(volume: np.ndarray, mode: str, window: tuple[float, float]):
    volume = np.nan_to_num(volume.astype(np.float32), nan=-1000.0,
                           posinf=window[1], neginf=window[0])
    if mode == "window":
        low, high = window
        volume = np.clip(volume, low, high)
        return (volume - low) / max(high - low, 1e-6)
    low, high = float(volume.min()), float(volume.max())
    return (volume - low) / max(high - low, 1e-6)


def prepare_row(row, dicom_root: Path, min_slices: int, min_coverage_mm: float,
                normalization: str, margin: int):
    series_dir = find_series_dir(dicom_root, str(row.patient_id), str(row.study_uid), str(row.series_uid))
    slices = read_dicom_slices(series_dir)
    slices, _ = filter_localizers(slices)
    slices, _ = filter_slices_by_dominant_shape(slices)
    slices, row_cos, col_cos, positions = sort_slices(slices)
    slices, _ = deduplicate_identical_positions(slices, positions)
    slices, row_cos, col_cos, positions = sort_slices(slices)
    validate_geometry(slices, positions, min_slices, min_coverage_mm, 1.5)
    # stack_hu returns (H,W,D); transpose to (D,H,W) for RadJEPA windows.
    volume = np.transpose(stack_hu(slices), (2, 0, 1))
    source_shape = volume.shape
    volume, bounds = crop_foreground(volume, margin=margin)
    volume = resize_xy(volume)
    volume = normalize(volume, normalization, (-1000.0, 400.0))
    return volume, source_shape, bounds, str(series_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--dicom-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--output-manifest', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--normalization', choices=['window', 'minmax'], default='window')
    ap.add_argument('--foreground-margin', type=int, default=8)
    ap.add_argument('--min-slices', type=int, default=50)
    ap.add_argument('--min-coverage-mm', type=float, default=200.0)
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    if args.limit:
        manifest = manifest.head(args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in manifest.iterrows():
        try:
            volume, source_shape, bounds, dicom_dir = prepare_row(
                row, args.dicom_root, args.min_slices, args.min_coverage_mm,
                args.normalization, args.foreground_margin)
            out = args.output_root / str(row.patient_id) / f"{row.patient_id}_{row.timepoint}_{row.series_uid}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, volume[None].astype(np.float32))
            result = row.to_dict()
            result.update({'npy_path': str(out), 'npy_shape': 'x'.join(map(str, (1, *volume.shape))),
                           'source_shape_dhw': 'x'.join(map(str, source_shape)),
                           'foreground_bounds_dhw': 'x'.join(map(str, bounds)),
                           'normalization': args.normalization, 'status': 'prepared',
                           'dicom_dir_resolved': dicom_dir})
            print(f'[{i+1}/{len(manifest)}] {out} shape={result["npy_shape"]}')
        except Exception as exc:
            result = row.to_dict(); result.update({'npy_path': '', 'status': 'error', 'error': repr(exc)})
            print(f'[{i+1}/{len(manifest)}] ERROR {exc}')
        rows.append(result)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with args.output_manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
