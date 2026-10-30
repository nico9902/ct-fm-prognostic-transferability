#!/usr/bin/env python3
"""E4 native-grid token geometry for attention masking, plus DINOv3 regional weights.

For each of the five encoders this maps, on the encoder's *native* token grid
(before any 8^3 pooling), which tokens touch the GTV and which tokens form a
count-matched same-shape control. Both masks come from voxel centers on the
original CT grid: the GTV resampled nearest from the cached 1 mm mask, and that
same mask translated by the locked control shift.

It also writes the DINOv3 saved-grid (8^3) GTV, input-sample, and 0-20 mm
shell weights used by the E4 regional probes. DINOv3's saved tokens are an
adaptive average of its (slices, 14, 14) native grid, so its saved-grid weights
are the same adaptive average of native footprint fractions.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

import audit_lung1_e4_gtv_token_mapping as gtv_mapping
from audit_lung1_e4_gtv_token_mapping import (
    COHORT, MASKS, ROOT, empty_reference, footprint_weights, init_worker,
    source_reference, summarize_tokens,
)
from audit_lung1_e4_peritumor_mapping import inside_reference, ring_points_xyz


MODELS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
CONTROLS = ROOT / "outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv"
TOKENS = ROOT / "outputs/lung1_e2_tokens"
OUTPUT = ROOT / "outputs/lung1_e4_token_masking/geometry"
DINOV3_OUTPUT = ROOT / "outputs/lung1_e4_geometry_dinov3"
DINOV3_SIZE, DINOV3_PATCH, DINOV3_SLICE_MM = 224, 16, 3.0
# Native token grids in each encoder's token axis order (COLIPRI XYZ, others ZYX).
NATIVE_GRIDS = {"colipri": (24, 24, 24), "tangerine": (16, 16, 16),
                "radjepa": (4, 16, 16), "ctclip": (24, 24, 24)}


def dinov3_reference(native: sitk.Image) -> sitk.Image:
    """DINOv3 input grid: LPS, linear z resample to 3 mm, in-plane resize to 224."""
    oriented = sitk.DICOMOrient(native, "LPS")
    size = np.asarray(oriented.GetSize(), dtype=int)
    spacing = np.asarray(oriented.GetSpacing(), dtype=float)
    direction = np.asarray(oriented.GetDirection(), dtype=float).reshape(3, 3)
    slices = max(1, int(round(size[2] * spacing[2] / DINOV3_SLICE_MM)))
    target_size = np.asarray((DINOV3_SIZE, DINOV3_SIZE, slices), dtype=int)
    ratio = size / target_size
    # interpolate(..., align_corners=False): output center j maps to input (j + 0.5) * ratio - 0.5.
    origin = np.asarray(oriented.GetOrigin()) + direction @ (spacing * (0.5 * ratio - 0.5))
    return empty_reference(tuple(target_size), spacing * ratio, origin, direction)


def input_reference(model: str, native: sitk.Image) -> sitk.Image:
    if model == "dinov3":
        reference = dinov3_reference(native)
        return sitk.Resample(sitk.DICOMOrient(native, "LPS"), reference, sitk.Transform(),
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return gtv_mapping.input_mask(model, native)


def native_grid(model: str, reference: sitk.Image) -> tuple[int, int, int]:
    if model == "dinov3":
        return (int(reference.GetSize()[2]), DINOV3_SIZE // DINOV3_PATCH, DINOV3_SIZE // DINOV3_PATCH)
    return NATIVE_GRIDS[model]


def physical_points(image: sitk.Image, points_xyz: np.ndarray) -> np.ndarray:
    origin = np.asarray(image.GetOrigin(), dtype=float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return origin + (points_xyz * spacing) @ direction.T


def token_centers_mm(reference: sitk.Image, grid: tuple[int, int, int], model: str) -> np.ndarray:
    """Physical centers of native token blocks, flattened in token order."""
    shape = np.asarray(reference.GetSize(), dtype=int)
    if model != "colipri":
        shape = shape[::-1]
    block = shape // np.asarray(grid)
    index = np.stack(np.meshgrid(*[np.arange(g) for g in grid], indexing="ij"), -1).reshape(-1, 3)
    voxel = index * block + (block - 1) / 2
    if model != "colipri":
        voxel = voxel[:, ::-1]
    return physical_points(reference, voxel.astype(float))


def matched_control(control_fp: np.ndarray, gtv_tokens: np.ndarray,
                    centers: np.ndarray) -> np.ndarray:
    """Exactly as many control tokens as GTV tokens, never overlapping the GTV.

    Control-touching tokens are taken by decreasing control footprint; any
    remaining slots go to the nearest non-GTV tokens to the control centroid.
    """
    n_target = int(gtv_tokens.sum())
    weights = control_fp.reshape(-1)
    centroid = (weights[:, None] * centers).sum(0) / weights.sum()
    distance = np.linalg.norm(centers - centroid, axis=1)
    order = np.lexsort((distance, -weights))
    order = order[~gtv_tokens.reshape(-1)[order]]
    chosen = np.zeros(len(weights), dtype=bool)
    chosen[order[:n_target]] = True
    if chosen.sum() != n_target:
        raise ValueError("Not enough non-GTV tokens for a count-matched control")
    return chosen


def native_footprint(mask: sitk.Image, points_xyz: np.ndarray, reference: sitk.Image,
                     grid: tuple[int, int, int], model: str) -> tuple[float, np.ndarray]:
    """Same mapping as footprint_weights, without its 1.01 sanity bound.

    Native tokens can be thinner than the source slice spacing resolves, so the
    voxel-center count can exceed one token volume; the fraction is capped at 1.
    Only footprint > 0 defines a masked token.
    """
    continuous = ((physical_points(mask, points_xyz) - np.asarray(reference.GetOrigin()))
                  @ np.linalg.inv(np.asarray(reference.GetDirection()).reshape(3, 3)).T
                  ) / np.asarray(reference.GetSpacing())
    size = np.asarray(reference.GetSize(), dtype=float)
    inside = np.all((continuous >= -0.5) & (continuous < size - 0.5), axis=1)
    shape = np.asarray(reference.GetSize(), dtype=int)
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    if model != "colipri":
        shape, continuous, spacing = shape[::-1], continuous[:, ::-1], spacing[::-1]
    grid_array = np.asarray(grid, dtype=int)
    if np.any(shape % grid_array):
        raise ValueError(f"Input shape {shape} is not divisible by grid {grid}")
    block = shape // grid_array
    indices = np.clip(np.floor((continuous[inside] + 0.5) / block).astype(int), 0, grid_array - 1)
    counts = np.bincount(np.ravel_multi_index(indices.T, grid), minlength=int(np.prod(grid)))
    fraction = counts * (np.prod(mask.GetSpacing()) / np.prod(block * spacing))
    return float(inside.mean()), np.minimum(fraction, 1).astype(np.float32).reshape(grid)


def adaptive_pool(values: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float64))[None, None]
    return F.adaptive_avg_pool3d(tensor, grid)[0, 0].numpy()


def one_patient(row: dict, saved_grid_dinov3: tuple[int, int, int]) -> tuple:
    patient_id = row["patient_id"]
    folder = MASKS / patient_id
    metadata = json.loads((folder / f"{patient_id}_metadata.json").read_text())
    source = source_reference(metadata)
    mask = sitk.ReadImage(str(folder / f"{patient_id}_gtv_mask.nii.gz"), sitk.sitkUInt8)
    native = sitk.Resample(mask, source, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    native_array = sitk.GetArrayFromImage(native) > 0
    gtv_points = np.argwhere(native_array)[:, ::-1].astype(np.float32)
    if not len(gtv_points):
        raise ValueError(f"{patient_id}: empty GTV on original CT grid")
    has_control = pd.notna(row["sham_dx_voxels"])
    if has_control:
        dx, dy = int(row["sham_dx_voxels"]), int(row["sham_dy_voxels"])
        control_points = gtv_points + np.asarray((dx, dy, 0), dtype=np.float32)
        z, y, x = np.argwhere(native_array).T
        if native_array[z, y + dy, x + dx].any():
            raise ValueError(f"{patient_id}: locked control overlaps GTV")
    records, arrays = [], {}
    for model in MODELS:
        reference = input_reference(model, native)
        grid = native_grid(model, reference)
        fov, gtv_fp = native_footprint(native, gtv_points, reference, grid, model)
        gtv_tokens = gtv_fp > 0
        record = {"patient_id": patient_id, "model": model, "native_grid": list(grid),
                  "n_native_tokens": int(np.prod(grid)),
                  "gtv_center_fraction_in_input_fov": fov,
                  "n_gtv_tokens": int(gtv_tokens.sum()),
                  "gtv_token_fraction": float(gtv_tokens.mean()),
                  "gtv_fraction_of_masked_volume": float(gtv_fp[gtv_tokens].sum() / gtv_tokens.sum())}
        if not gtv_tokens.any():
            raise ValueError(f"{patient_id}/{model}: no native GTV token")
        out = {"gtv_footprint": gtv_fp.astype(np.float32), "gtv": gtv_tokens}
        if has_control:
            control_fov, control_fp = native_footprint(native, control_points, reference, grid, model)
            if control_fp.sum() <= 0:
                raise ValueError(f"{patient_id}/{model}: control outside the input field")
            control = matched_control(control_fp, gtv_tokens,
                                      token_centers_mm(reference, grid, model)).reshape(grid)
            record |= {"control_center_fraction_in_input_fov": control_fov,
                       "n_control_touching_tokens": int((control_fp > 0).sum()),
                       "n_control_touching_overlapping_gtv": int(((control_fp > 0) & gtv_tokens).sum()),
                       "n_control_tokens": int(control.sum()),
                       "control_footprint_captured": float(
                           control_fp[control].sum() / control_fp.sum())}
            out |= {"control_footprint": control_fp.astype(np.float32), "control": control}
        records.append(record)
        arrays[model] = out
    # DINOv3 saved-grid regional weights, using the cached 1 mm GTV like the other models.
    reference = input_reference("dinov3", native)
    grid = native_grid("dinov3", reference)
    mm_points = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)[:, ::-1].astype(np.float32)
    fov, footprint = native_footprint(mask, mm_points, reference, grid, "dinov3")
    ring = ring_points_xyz(mask)
    ring = ring[inside_reference(mask, ring, source)]
    ring_fov, ring_fp = native_footprint(mask, ring, reference, grid, "dinov3")
    sampled, stats = summarize_tokens(sitk.GetArrayFromImage(reference), (grid[0], 14, 14))
    tumor = adaptive_pool(footprint, saved_grid_dinov3).reshape(-1)
    peri = adaptive_pool(ring_fp, saved_grid_dinov3).reshape(-1)
    regional = {"weights": tumor.astype(np.float32), "peri20": peri.astype(np.float32),
                "input_sample": adaptive_pool(sampled, saved_grid_dinov3).reshape(-1).astype(np.float32)}
    span = (np.asarray(reference.GetSpacing())[::-1] *
            np.asarray(stats["token_block_shape_token_axes"])).tolist()
    regional_record = {
        "patient_id": patient_id, "model": "dinov3", "saved_grid": list(saved_grid_dinov3),
        "native_grid": list(grid), "token_axis_order": "ZYX",
        "native_token_span_model_axes_mm": span,
        "gtv_center_fraction_in_input_fov": fov,
        "n_tokens_any_gtv_footprint": int((tumor > 0).sum()),
        "n_tokens_gtv_ge_10pct_footprint": int((tumor >= 0.10).sum()),
        "n_tokens_gtv_majority_footprint": int((tumor >= 0.50).sum()),
        "max_token_gtv_fraction_footprint": float(tumor.max()),
        "n_tokens_any_gtv": int((regional["input_sample"] > 0).sum()),
        "tumor_input_voxels": stats["tumor_input_voxels"],
        "ring_center_fraction_in_input_fov": ring_fov,
        "n_tokens_any_peri20": int((peri > 0).sum()),
        "n_tokens_peri20_only": int(((peri > 0) & (tumor == 0)).sum()),
    }
    return records, arrays, regional_record, regional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    control = pd.read_csv(CONTROLS, dtype={"patient_id": str})
    data = data.merge(control[["patient_id", "sham_dx_voxels", "sham_dy_voxels"]],
                      on="patient_id", how="left", validate="one_to_one")
    manifest = pd.read_csv(TOKENS / "dinov3/original_ct/token_manifest.csv", dtype={"patient_id": str})
    if set(manifest.patient_id) != set(data.patient_id):
        raise ValueError("DINOv3 token manifest does not match cohort")
    saved_grid = tuple(json.loads(manifest.saved_grid.iloc[0]))
    native_by_patient = manifest.set_index("patient_id").native_grid.map(json.loads).to_dict()
    for output, protocol in (
            (OUTPUT, {"version": 1, "models": list(MODELS),
                      "native_grids_fixed": {k: list(v) for k, v in NATIVE_GRIDS.items()},
                      "dinov3_native_grid": "slices_at_3mm x 14 x 14",
                      "points": "original-CT voxel centers of the nearest-resampled cached GTV; control = same points shifted by the locked control (dx, dy)",
                      "gtv_tokens": "native tokens with any GTV footprint",
                      "control_tokens": "count-matched to GTV tokens: control-touching tokens by decreasing footprint, then nearest non-GTV tokens to the control centroid; never a GTV token"}),
            (DINOV3_OUTPUT, {"version": 1, "model": "dinov3", "saved_grid": list(saved_grid),
                             "token_weight": "cached 1 mm GTV physical footprint per native (slice, 16x16 patch) token, adaptive-average pooled to the saved grid like the tokens",
                             "peri20_weight": "same for the (0,20] mm shell within source CT and input field"})):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError(f"{path}: saved protocol differs")
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = data.to_dict("records")[:args.max_patients or None]
    missing = [row for row in rows if not (OUTPUT / "patients" / f"{row['patient_id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for i, (row, result) in enumerate(zip(missing, pool.map(
                partial(one_patient, saved_grid_dinov3=saved_grid), missing)), 1):
            pid = row["patient_id"]
            records, arrays, regional_record, regional = result
            dinov3_native = next(r["native_grid"] for r in records if r["model"] == "dinov3")
            if dinov3_native != native_by_patient[pid]:
                raise ValueError(f"{pid}: DINOv3 native grid {dinov3_native} differs from token cache")
            for model, values in arrays.items():
                path = OUTPUT / "masks" / model / f"{pid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **values)
            for name, folder in (("weights", "weights"), ("peri20", "peri20/weights"),
                                 ("input_sample", "input_sample_weights")):
                path = DINOV3_OUTPUT / folder / "dinov3" / f"{pid}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, regional[name])
            for folder, value in ((DINOV3_OUTPUT, [regional_record]), (OUTPUT, records)):
                path = folder / "patients" / f"{pid}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, indent=2) + "\n")
            if i % 10 == 0 or i == len(missing):
                print(f"Mapped {len(rows) - len(missing) + i}/{len(rows)} patients", flush=True)
    if len(rows) == len(data):
        for folder in (OUTPUT, DINOV3_OUTPUT):
            frame = pd.DataFrame([r for pid in data.patient_id for r in json.loads(
                (folder / "patients" / f"{pid}.json").read_text())])
            frame.to_csv(folder / "patient_model_geometry.csv", index=False)
            (folder / "COMPLETE").write_text(f"{len(data)} patients\n")
        # The regional Cox runner checks for the peri20 completion marker too.
        (DINOV3_OUTPUT / "peri20").mkdir(exist_ok=True)
        frame.to_csv(DINOV3_OUTPUT / "peri20/patient_model_geometry.csv", index=False)
        (DINOV3_OUTPUT / "peri20/protocol.json").write_text(
            (DINOV3_OUTPUT / "protocol.json").read_text())
        (DINOV3_OUTPUT / "peri20/COMPLETE").write_text(f"{len(data)} patients\n")


if __name__ == "__main__":
    main()
