#!/usr/bin/env python3
"""Choose a same-shape, same-axial-level tumor-free control mask for each LUNG1 CT."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from extract_lung1_e1_embeddings import COHORT, ROOT, read_image, sha256


MASKS = ROOT / "data/processed/lung1/thorax_hu"
OUTPUT = ROOT / "outputs/lung1_e4_masked_volume"
SHIFTS_MM = (35, 55, 75, 100, 125, 150, 180, 210)


def one_patient(row: dict) -> dict:
    patient_id = row["patient_id"]
    ct = read_image(type("Row", (), row)(), "original_ct", MASKS)
    mask = sitk.ReadImage(str(MASKS / patient_id / f"{patient_id}_gtv_mask.nii.gz"), sitk.sitkUInt8)
    lung_mask = sitk.ReadImage(str(MASKS / patient_id / f"{patient_id}_lung_mask.nii.gz"), sitk.sitkUInt8)
    gtv = sitk.GetArrayFromImage(sitk.Resample(mask, ct, sitk.Transform(),
                                               sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
    lung = sitk.GetArrayFromImage(sitk.Resample(lung_mask, ct, sitk.Transform(),
                                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
    if not gtv.any():
        raise ValueError(f"{patient_id}: GTV absent from source CT grid")
    volume = sitk.GetArrayViewFromImage(ct)
    points = np.argwhere(gtv)
    z, y, x = points.T
    gtv_median = float(np.median(volume[z, y, x]))
    spacing = ct.GetSpacing()
    candidates = []
    seen = set()
    for distance in SHIFTS_MM:
        for dx_mm, dy_mm in ((distance, 0), (-distance, 0), (0, distance), (0, -distance),
                            (distance, distance), (distance, -distance),
                            (-distance, distance), (-distance, -distance)):
            dx = int(round(dx_mm / spacing[0]))
            dy = int(round(dy_mm / spacing[1]))
            if (dx, dy) in seen or (dx == 0 and dy == 0):
                continue
            seen.add((dx, dy))
            shifted_x, shifted_y = x + dx, y + dy
            if (shifted_x.min() < 0 or shifted_x.max() >= gtv.shape[2] or
                    shifted_y.min() < 0 or shifted_y.max() >= gtv.shape[1]):
                continue
            if gtv[z, shifted_y, shifted_x].any():
                continue
            values = volume[z, shifted_y, shifted_x]
            body_fraction = float(np.mean(values > -950))
            lung_fraction = float(np.mean(lung[z, shifted_y, shifted_x]))
            if body_fraction < 0.95 or lung_fraction < 0.30:
                continue
            candidate_median = float(np.median(values))
            # Match local tissue intensity while preferring a nearby, nonoverlapping
            # site at the same axial levels. Image intensities alone set this rule.
            score = abs(candidate_median - gtv_median) + 0.05 * np.hypot(dx_mm, dy_mm)
            candidates.append((score, dx, dy, candidate_median, body_fraction, lung_fraction))
    if not candidates:
        return {"patient_id": patient_id, "ct_series_uid": row["ct_series_uid"],
                "gtv_source_ct_voxels": int(gtv.sum()), "gtv_median_hu": gtv_median,
                "sham_dx_voxels": None, "sham_dy_voxels": None,
                "sham_median_hu": None, "sham_hu_median_gap": None,
                "sham_body_fraction": None, "sham_lung_fraction": None,
                "n_candidates": 0}
    _, dx, dy, median, body_fraction, lung_fraction = min(
        candidates, key=lambda item: (item[0], item[1], item[2]))
    return {"patient_id": patient_id, "ct_series_uid": row["ct_series_uid"],
            "gtv_source_ct_voxels": int(gtv.sum()), "gtv_median_hu": gtv_median,
            "sham_dx_voxels": int(dx), "sham_dy_voxels": int(dy),
            "sham_median_hu": median, "sham_hu_median_gap": abs(median - gtv_median),
            "sham_body_fraction": body_fraction, "sham_lung_fraction": lung_fraction,
            "n_candidates": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    output = OUTPUT / "control_masks"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"cohort_sha256": sha256(COHORT), "source": "original_ct",
                "mask_source": "cached 1 mm GTV resampled nearest to original CT",
                "shifts_mm": list(SHIFTS_MM), "same_axial_levels": True,
                "selection": "minimum absolute median HU difference plus 0.05 HU/mm translation penalty; no GTV overlap; at least 95% CT voxels above -950 HU and 30% R231 lung voxels",
                "outcome_independent": True}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Control-mask protocol differs")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = data.to_dict("records")[:args.max_patients or None]
    missing = [row for row in rows if not (output / "patients" / f"{row['patient_id']}.json").exists()]
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(one_patient, missing), 1):
            p = output / "patients" / f"{result['patient_id']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=2) + "\n")
            if i % 10 == 0 or i == len(missing):
                print(f"Prepared control {len(rows)-len(missing)+i}/{len(rows)}", flush=True)
    if len(rows) == len(data):
        frame = pd.DataFrame([json.loads((output / "patients" / f"{pid}.json").read_text())
                              for pid in data.patient_id])
        frame.to_csv(output / "control_manifest.csv", index=False)
        (output / "COMPLETE").write_text(f"{len(data)} patients\n")


if __name__ == "__main__":
    main()
