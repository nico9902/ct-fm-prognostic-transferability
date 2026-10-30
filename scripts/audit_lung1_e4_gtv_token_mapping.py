#!/usr/bin/env python3
"""Map LUNG1 GTV masks through E2 original-CT preprocessing to saved token grids."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
MASKS = ROOT / "data/processed/lung1/thorax_hu"
TOKENS = ROOT / "outputs/lung1_e2_tokens"
OUTPUT = ROOT / "outputs/lung1_e4_geometry"
MODELS = ("colipri", "tangerine", "radjepa", "ctclip")
COLIPRI_PROCESSOR = None


def init_worker() -> None:
    global COLIPRI_PROCESSOR
    from colipri import get_processor
    COLIPRI_PROCESSOR = get_processor(image_only=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def source_reference(metadata: dict) -> sitk.Image:
    direction = np.asarray(metadata["crop_direction_lps"], dtype=float).reshape(3, 3)
    start_xyz = np.asarray([metadata["bbox_zyx"][2][0], metadata["bbox_zyx"][1][0],
                            metadata["bbox_zyx"][0][0]], dtype=float)
    origin = np.asarray(metadata["crop_origin_lps_mm"], dtype=float) - direction @ start_xyz
    reference = sitk.Image([int(x) for x in metadata["source_size_xyz"]], sitk.sitkUInt8)
    reference.SetOrigin(tuple(origin))
    reference.SetDirection(tuple(direction.ravel()))
    reference.SetSpacing(tuple(float(x) for x in metadata["source_spacing_xyz_mm"]))
    return reference


def empty_reference(size: tuple[int, int, int], spacing: np.ndarray,
                    origin: np.ndarray, direction: np.ndarray) -> sitk.Image:
    ref = sitk.Image([int(x) for x in size], sitk.sitkUInt8)
    ref.SetSpacing(tuple(float(x) for x in spacing))
    ref.SetOrigin(tuple(float(x) for x in origin))
    ref.SetDirection(tuple(float(x) for x in direction.ravel()))
    return ref


def input_mask(model: str, native: sitk.Image) -> sitk.Image:
    size = np.asarray(native.GetSize(), dtype=int)
    spacing = np.asarray(native.GetSpacing(), dtype=float)
    origin = np.asarray(native.GetOrigin(), dtype=float)
    direction = np.asarray(native.GetDirection(), dtype=float).reshape(3, 3)
    if model == "tangerine":
        target_size = np.full(3, 256, dtype=int)
        target = empty_reference(tuple(target_size), size * spacing / target_size,
                                 origin, direction)
        return sitk.Resample(native, target, sitk.Transform(), sitk.sitkNearestNeighbor,
                             0, sitk.sitkUInt8)
    if model == "radjepa":
        oriented = sitk.DICOMOrient(native, "RAS")
        size = np.asarray(oriented.GetSize(), dtype=int)
        target_size = np.asarray((256, 256, 32), dtype=int)
        # scipy.ndimage.zoom uses grid_mode=False: first and last voxel centers align.
        target_spacing = np.asarray(oriented.GetSpacing()) * (size - 1) / (target_size - 1)
        target = empty_reference(tuple(target_size), target_spacing,
                                 np.asarray(oriented.GetOrigin()),
                                 np.asarray(oriented.GetDirection()).reshape(3, 3))
        return sitk.Resample(oriented, target, sitk.Transform(), sitk.sitkNearestNeighbor,
                             0, sitk.sitkUInt8)
    if model == "ctclip":
        oriented = sitk.DICOMOrient(native, "LPS")
        size = np.asarray(oriented.GetSize(), dtype=int)
        spacing = np.asarray(oriented.GetSpacing(), dtype=float)
        direction = np.asarray(oriented.GetDirection(), dtype=float).reshape(3, 3)
        intermediate_size = np.maximum(1, (size * spacing /
                                           np.asarray((0.75, 0.75, 1.5))).astype(int))
        ratio = size / intermediate_size
        intermediate_spacing = spacing * ratio
        # PyTorch interpolate(..., align_corners=False) maps output voxel centers
        # to input index (j + 0.5) * ratio - 0.5.
        intermediate_origin = np.asarray(oriented.GetOrigin()) + direction @ (
            spacing * (0.5 * ratio - 0.5))
        final_size = np.asarray((480, 480, 240), dtype=int)
        crop_start = np.maximum((intermediate_size - final_size) // 2, 0)
        pad_before = np.maximum((final_size - intermediate_size) // 2, 0)
        final_origin = intermediate_origin + direction @ (
            (crop_start - pad_before) * intermediate_spacing)
        target = empty_reference(tuple(final_size), intermediate_spacing,
                                 final_origin, direction)
        return sitk.Resample(oriented, target, sitk.Transform(), sitk.sitkNearestNeighbor,
                             0, sitk.sitkUInt8)
    if model == "colipri":
        import torchio as tio
        if COLIPRI_PROCESSOR is None:
            raise RuntimeError("COLIPRI processor was not initialized")
        transformed = COLIPRI_PROCESSOR._image_transform(tio.LabelMap.from_sitk(native))
        return transformed.as_sitk()
    raise ValueError(model)


def footprint_weights(mask: sitk.Image, source_points_xyz: np.ndarray,
                      target: sitk.Image, grid: tuple[int, int, int],
                      model: str) -> tuple[float, np.ndarray]:
    source_origin = np.asarray(mask.GetOrigin(), dtype=float)
    source_spacing = np.asarray(mask.GetSpacing(), dtype=float)
    source_direction = np.asarray(mask.GetDirection(), dtype=float).reshape(3, 3)
    physical = source_origin + (source_points_xyz * source_spacing) @ source_direction.T
    target_origin = np.asarray(target.GetOrigin(), dtype=float)
    target_spacing = np.asarray(target.GetSpacing(), dtype=float)
    target_direction = np.asarray(target.GetDirection(), dtype=float).reshape(3, 3)
    continuous = ((physical - target_origin) @ np.linalg.inv(target_direction).T) / target_spacing
    size = np.asarray(target.GetSize(), dtype=float)
    inside = np.all((continuous >= -0.5) & (continuous < size - 0.5), axis=1)
    grid_array = np.asarray(grid, dtype=int)
    shape = np.asarray(target.GetSize(), dtype=int)
    if model != "colipri":
        shape = shape[::-1]
        continuous = continuous[:, ::-1]
    if np.any(shape % grid_array):
        raise ValueError(f"Input shape {shape} is not divisible by grid {grid}")
    block = shape // grid_array
    indices = np.floor((continuous[inside] + 0.5) / block).astype(int)
    indices = np.clip(indices, 0, grid_array - 1)
    flat = np.ravel_multi_index(indices.T, grid)
    counts = np.bincount(flat, minlength=int(np.prod(grid)))
    voxel_volume = float(np.prod(source_spacing))
    token_volume = float(np.prod(block * target_spacing))
    fraction = counts.astype(np.float32) * (voxel_volume / token_volume)
    if np.any(fraction > 1.01):
        raise ValueError(f"{model}: impossible GTV footprint fraction {fraction.max()}")
    return float(inside.mean()), np.minimum(fraction, 1).reshape(grid)


def summarize_tokens(array: np.ndarray, grid: tuple[int, int, int]) -> tuple[np.ndarray, dict]:
    shape = np.asarray(array.shape, dtype=int)
    grid_array = np.asarray(grid, dtype=int)
    if np.any(shape % grid_array):
        raise ValueError(f"Input shape {shape} is not divisible by saved token grid {grid}")
    block = shape // grid_array
    grouped = array.reshape(grid[0], block[0], grid[1], block[1],
                            grid[2], block[2]).sum(axis=(1, 3, 5))
    fraction = grouped.astype(np.float32) / int(np.prod(block))
    return fraction, {"input_shape_token_axes": shape.tolist(),
                      "token_block_shape_token_axes": block.tolist(),
                      "tumor_input_voxels": int(array.sum()),
                      "n_tokens_any_gtv": int((fraction > 0).sum()),
                      "n_tokens_gtv_ge_1pct": int((fraction >= 0.01).sum()),
                      "n_tokens_gtv_ge_10pct": int((fraction >= 0.10).sum()),
                      "n_tokens_gtv_majority": int((fraction >= 0.50).sum()),
                      "max_token_gtv_fraction": float(fraction.max()),
                      "effective_gtv_tokens": float(fraction.sum())}


def audit_patient(patient_id: str, grids: dict[str, tuple[int, int, int]]) -> tuple[list[dict], dict]:
    folder = MASKS / patient_id
    metadata = json.loads((folder / f"{patient_id}_metadata.json").read_text())
    if metadata["patient_id"] != patient_id or metadata["mode"] != "thorax_hu":
        raise ValueError(f"{patient_id}: source mask metadata mismatch")
    source = source_reference(metadata)
    mask = sitk.ReadImage(str(folder / f"{patient_id}_gtv_mask.nii.gz"), sitk.sitkUInt8)
    if (tuple(mask.GetSize()) != tuple(metadata["crop_size_xyz"])
            or not np.allclose(mask.GetOrigin(), metadata["crop_origin_lps_mm"], atol=1e-5)):
        raise ValueError(f"{patient_id}: cached GTV geometry mismatch")
    cropped_array = sitk.GetArrayViewFromImage(mask)
    points = np.argwhere(cropped_array > 0)[:, ::-1].astype(np.float32)
    if not len(points):
        raise ValueError(f"{patient_id}: empty cached GTV")
    native = sitk.Resample(mask, source, sitk.Transform(), sitk.sitkNearestNeighbor,
                           0, sitk.sitkUInt8)
    if sitk.GetArrayViewFromImage(native).sum() == 0:
        raise ValueError(f"{patient_id}: empty GTV on original CT grid")
    records, weights = [], {}
    for model in MODELS:
        transformed = input_mask(model, native)
        array = sitk.GetArrayFromImage(transformed)
        if array.ndim != 3:
            raise ValueError(f"{patient_id}/{model}: not a 3-D mask")
        # TorchIO gives COLIPRI a C,X,Y,Z tensor. The other three encoders
        # receive C,Z,Y,X tensors. Match the saved-token flattening order.
        if model == "colipri":
            array = array.transpose(2, 1, 0)
        sampled_fraction, stats = summarize_tokens(array, grids[model])
        fov, footprint = footprint_weights(mask, points, transformed, grids[model], model)
        axis_spacing = np.asarray(transformed.GetSpacing())
        if model != "colipri":
            axis_spacing = axis_spacing[::-1]
        span = (axis_spacing * np.asarray(stats["token_block_shape_token_axes"])).tolist()
        record = {"patient_id": patient_id, "model": model,
                  "saved_grid": list(grids[model]),
                  "token_axis_order": "XYZ" if model == "colipri" else "ZYX",
                  "token_span_model_axes_mm": span,
                  "gtv_center_fraction_in_input_fov": fov,
                  "n_tokens_any_gtv_footprint": int((footprint > 0).sum()),
                  "n_tokens_gtv_ge_1pct_footprint": int((footprint >= 0.01).sum()),
                  "n_tokens_gtv_ge_10pct_footprint": int((footprint >= 0.10).sum()),
                  "n_tokens_gtv_majority_footprint": int((footprint >= 0.50).sum()),
                  "max_token_gtv_fraction_footprint": float(footprint.max()),
                  **stats}
        records.append(record)
        weights[model] = (footprint.reshape(-1), sampled_fraction.reshape(-1))
    return records, weights


def load_grids(data: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    grids = {}
    for model in MODELS:
        manifest = pd.read_csv(TOKENS / model / "original_ct/token_manifest.csv",
                               dtype={"patient_id": str, "ct_series_uid": str})
        if (manifest.patient_id.duplicated().any() or
                set(manifest.patient_id) != set(data.patient_id)):
            raise ValueError(f"{model}: token manifest does not match cohort")
        ordered = manifest.set_index("patient_id").loc[data.patient_id]
        if not np.array_equal(ordered.ct_series_uid.to_numpy(), data.ct_series_uid.to_numpy()):
            raise ValueError(f"{model}: token series UID differs from cohort")
        values = {value for value in manifest.saved_grid}
        if len(values) != 1:
            raise ValueError(f"{model}: multiple saved token grids")
        grid = tuple(int(x) for x in json.loads(next(iter(values))))
        if not (manifest.n_tokens.astype(int) == np.prod(grid)).all():
            raise ValueError(f"{model}: manifest token count does not match grid")
        grids[model] = grid
    return grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-patients", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_patients < 0:
        parser.error("workers must be positive and max-patients nonnegative")
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    grids = load_grids(data)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": 3,
                "source": "cached 1 mm GTV masks from original CT SEG",
                "target": "E2 original-CT preprocessing and saved token grids",
                "models": list(MODELS), "saved_grids": {k: list(v) for k, v in grids.items()},
                "token_weight": "1 mm GTV voxel-center physical footprint in each saved token block, divided by token physical volume; weights/<model>/*.npy",
                "input_sample_weight": "fraction of nearest-neighbor sampled input mask voxels per token block; input_sample_weights/<model>/*.npy",
                "fov_retention": "fraction of cached 1 mm GTV voxel centers inside final input image field"}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Saved geometry protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    ids = data.patient_id.tolist()[:args.max_patients or None]
    missing = [pid for pid in ids if not (output / "patients" / f"{pid}.json").exists()]
    worker = partial(audit_patient, grids=grids)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for i, (pid, result) in enumerate(zip(missing, pool.map(worker, missing)), 1):
            records, weights = result
            for model, (footprint, sampled) in weights.items():
                path = output / "weights" / model / f"{pid}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, footprint)
                path = output / "input_sample_weights" / model / f"{pid}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, sampled)
            path = output / "patients" / f"{pid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(records, indent=2) + "\n")
            if i % 10 == 0 or i == len(missing):
                print(f"Mapped {len(ids) - len(missing) + i}/{len(ids)} patients", flush=True)
    if len(ids) == len(data):
        rows = [row for pid in ids for row in json.loads(
            (output / "patients" / f"{pid}.json").read_text())]
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "patient_model_geometry.csv", index=False)
        summary = []
        for model in MODELS:
            part = frame.loc[frame.model.eq(model)]
            spans = np.asarray(part.token_span_model_axes_mm.tolist(), dtype=float)
            coverage = part.gtv_center_fraction_in_input_fov.to_numpy(dtype=float)
            tokens = part.n_tokens_any_gtv_footprint.to_numpy(dtype=int)
            summary.append({"model": model, "n_patients": len(part),
                            "saved_grid": str(part.saved_grid.iloc[0]),
                            "median_token_span_model_axes_mm": json.dumps(
                                np.median(spans, axis=0).round(2).tolist()),
                            "min_gtv_center_fraction_in_fov": float(coverage.min()),
                            "n_fov_below_95pct": int((coverage < 0.95).sum()),
                            "n_fov_zero": int((coverage == 0).sum()),
                            "median_tokens_any_gtv_footprint": float(np.median(tokens)),
                            "n_zero_gtv_footprint_tokens": int((tokens == 0).sum()),
                            "n_zero_gtv_input_sample_tokens": int(
                                (part.n_tokens_any_gtv == 0).sum()),
                            "n_no_footprint_token_ge_10pct_gtv": int(
                                (part.n_tokens_gtv_ge_10pct_footprint == 0).sum()),
                            "n_no_majority_gtv_footprint_token": int(
                                (part.n_tokens_gtv_majority_footprint == 0).sum()),
                            "median_max_token_gtv_footprint_fraction": float(
                                part.max_token_gtv_fraction_footprint.median())})
        pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
        (output / "COMPLETE").write_text(f"{len(data)} patients x {len(MODELS)} models\n")


if __name__ == "__main__":
    main()
