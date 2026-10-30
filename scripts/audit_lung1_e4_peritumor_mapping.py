#!/usr/bin/env python3
"""Map the physical 0–20 mm GTV shell onto E4 cached token grids."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from audit_lung1_e4_gtv_token_mapping import (
    COHORT, MASKS, MODELS, OUTPUT, footprint_weights,
    init_worker, input_mask, load_grids, source_reference,
)


def ring_points_xyz(mask: sitk.Image, radius_mm: int = 20) -> np.ndarray:
    binary = sitk.GetArrayFromImage(mask) > 0
    positive = np.argwhere(binary)
    if not len(positive):
        raise ValueError("Empty GTV")
    lower = positive.min(axis=0) - radius_mm
    upper = positive.max(axis=0) + radius_mm + 1
    shape = upper - lower
    local = np.zeros(tuple(shape), dtype=bool)
    coordinates = positive - lower
    local[tuple(coordinates.T)] = True
    distance = distance_transform_edt(~local, sampling=mask.GetSpacing()[::-1])
    ring = np.argwhere((distance > 0) & (distance <= radius_mm))
    return (ring + lower)[:, ::-1].astype(np.float32)


def inside_reference(mask: sitk.Image, points_xyz: np.ndarray,
                     reference: sitk.Image) -> np.ndarray:
    origin = np.asarray(mask.GetOrigin(), dtype=float)
    spacing = np.asarray(mask.GetSpacing(), dtype=float)
    direction = np.asarray(mask.GetDirection(), dtype=float).reshape(3, 3)
    physical = origin + (points_xyz * spacing) @ direction.T
    ref_origin = np.asarray(reference.GetOrigin(), dtype=float)
    ref_spacing = np.asarray(reference.GetSpacing(), dtype=float)
    ref_direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    continuous = ((physical - ref_origin) @ np.linalg.inv(ref_direction).T) / ref_spacing
    size = np.asarray(reference.GetSize(), dtype=float)
    return np.all((continuous >= -0.5) & (continuous < size - 0.5), axis=1)


def one_patient(patient_id: str, grids: dict[str, tuple[int, int, int]]) -> tuple[list[dict], dict]:
    folder = MASKS / patient_id
    metadata = json.loads((folder / f"{patient_id}_metadata.json").read_text())
    source = source_reference(metadata)
    mask = sitk.ReadImage(str(folder / f"{patient_id}_gtv_mask.nii.gz"), sitk.sitkUInt8)
    points = ring_points_xyz(mask)
    points = points[inside_reference(mask, points, source)]
    if not len(points):
        raise ValueError(f"{patient_id}: empty 0–20 mm ring inside original CT")
    native = sitk.Resample(mask, source, sitk.Transform(), sitk.sitkNearestNeighbor,
                           0, sitk.sitkUInt8)
    records, weights = [], {}
    for model in MODELS:
        transformed = input_mask(model, native)
        coverage, fraction = footprint_weights(mask, points, transformed, grids[model], model)
        tumor = np.load(OUTPUT / "weights" / model / f"{patient_id}.npy", allow_pickle=False)
        flat = fraction.reshape(-1)
        if len(flat) != len(tumor):
            raise ValueError(f"{patient_id}/{model}: tumor/peritumor grid mismatch")
        records.append({"patient_id": patient_id, "model": model,
                        "ring_voxels_inside_original_ct": len(points),
                        "ring_center_fraction_in_input_fov": coverage,
                        "n_tokens_any_peri20": int((flat > 0).sum()),
                        "n_tokens_peri20_only": int(((flat > 0) & (tumor == 0)).sum()),
                        "n_tokens_tumor_and_peri20": int(((flat > 0) & (tumor > 0)).sum()),
                        "max_token_peri20_fraction": float(flat.max())})
        weights[model] = flat
    return records, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    grids = load_grids(data)
    if not (OUTPUT / "COMPLETE").exists():
        raise ValueError("Run complete GTV token mapping first")
    output = OUTPUT / "peri20"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"radius_mm": 20, "region": "(0,20] mm Euclidean distance from 1 mm GTV",
                "volume_rule": "1 mm ring voxel centers within source CT and input field divided by token block volume",
                "saved_grids": {model: list(grid) for model, grid in grids.items()}}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Peritumor mapping protocol differs")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    ids = data.patient_id.tolist()[:args.max_patients or None]
    missing = [pid for pid in ids if not (output / "patients" / f"{pid}.json").exists()]
    from functools import partial
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for i, (pid, result) in enumerate(zip(missing, pool.map(partial(one_patient, grids=grids), missing)), 1):
            records, weights = result
            for model, array in weights.items():
                p = output / "weights" / model / f"{pid}.npy"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, array)
            p = output / "patients" / f"{pid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(records, indent=2) + "\n")
            if i % 10 == 0 or i == len(missing):
                print(f"Mapped peri20 {len(ids) - len(missing) + i}/{len(ids)} patients", flush=True)
    if len(ids) == len(data):
        rows = [row for pid in ids for row in json.loads(
            (output / "patients" / f"{pid}.json").read_text())]
        pd.DataFrame(rows).to_csv(output / "patient_model_geometry.csv", index=False)
        (output / "COMPLETE").write_text(f"{len(data)} patients x {len(MODELS)} models\n")


if __name__ == "__main__":
    main()
