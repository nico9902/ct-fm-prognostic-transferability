#!/usr/bin/env python3
"""Re-encode original, GTV-filled, and matched-control CT inputs with frozen FMs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torch

import audit_lung1_e4_gtv_token_mapping as geometry
from extract_lung1_e1_embeddings import (
    CHECKPOINTS, COHORT, DIMENSIONS, ROOT, load_encoder, sha256,
)


OUTPUT = ROOT / "outputs/lung1_e4_masked_volume"
INPUTS = ROOT / "outputs/lung1_e2_inputs"
CONDITIONS = ("original", "gtv_filled", "sham_filled")
BLACK_VALUES = {"colipri": -1.0, "ctclip": -1.0,
                "tangerine": 0.0, "radjepa": 0.0}


def shifted_mask(gtv: np.ndarray, dx: int, dy: int) -> np.ndarray:
    points = np.argwhere(gtv)
    shifted = np.zeros_like(gtv, dtype=np.uint8)
    z, y, x = points.T
    shifted[z, y + dy, x + dx] = 1
    if shifted.sum() != gtv.sum() or np.logical_and(shifted, gtv).any():
        raise ValueError("Invalid translated control mask")
    return shifted


def to_input_labels(model: str, native: sitk.Image, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[float, ...]]:
    transformed = geometry.input_mask(model, native)
    array = sitk.GetArrayFromImage(transformed)
    spacing = tuple(float(v) for v in transformed.GetSpacing())
    if model == "colipri":
        array = array.transpose(2, 1, 0)
    else:
        spacing = spacing[::-1]
    if array.shape != expected_shape:
        raise ValueError(f"{model}: mask shape {array.shape} differs from CT input {expected_shape}")
    return array, spacing


def to_input_mask(model: str, native: sitk.Image, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[float, ...]]:
    labels, spacing = to_input_labels(model, native, expected_shape)
    return labels > 0, spacing


def fill_mask(volume: np.ndarray, mask: np.ndarray,
              spacing: tuple[float, ...], method: str = "median",
              black_value: float | None = None) -> tuple[np.ndarray, dict]:
    if method == "black":
        if black_value is None:
            raise ValueError("black_value is required for black masking")
        replaced = volume.copy()
        replaced[mask] = black_value
        change = np.abs(replaced[mask] - volume[mask])
        return replaced, {"mask_voxels": int(mask.sum()), "fill_value": black_value,
                          "method": method, "changed_voxels": int((change > 1e-7).sum()),
                          "mean_abs_input_change": float(change.mean()) if len(change) else 0.0}
    if not mask.any():
        return volume.copy(), {"mask_voxels": 0, "fill_value": None,
                               "changed_voxels": 0, "mean_abs_input_change": 0.0}
    points = np.argwhere(mask)
    radius = np.ceil(22 / np.asarray(spacing)).astype(int)
    lower = np.maximum(points.min(axis=0) - radius, 0)
    upper = np.minimum(points.max(axis=0) + radius + 1, volume.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    local_mask = mask[region]
    original = volume[region]
    distance = distance_transform_edt(~local_mask, sampling=spacing)
    shell = (distance >= 5) & (distance <= 20)
    if not shell.any():
        raise ValueError("No local 5–20 mm shell for fill")
    fill = float(np.median(original[shell]))
    if method == "median":
        replacement = fill
    elif method == "inpaint":
        _, nearest = distance_transform_edt(local_mask, sampling=spacing,
                                            return_indices=True)
        extrapolated = original.copy()
        extrapolated[local_mask] = original[tuple(axis[local_mask] for axis in nearest)]
        replacement = gaussian_filter(extrapolated, sigma=2 / np.asarray(spacing),
                                      mode="nearest")
    else:
        raise ValueError(method)
    alpha = np.maximum(0, 1 - distance / 3).astype(np.float32)
    alpha[local_mask] = 1
    replaced = volume.copy()
    updated = original * (1 - alpha) + replacement * alpha
    replaced[region] = updated
    changed = np.abs(updated - original) > 1e-7
    return replaced, {"mask_voxels": int(mask.sum()),
                      "fill_value": fill if method == "median" else None,
                      "method": method,
                      "changed_voxels": int(changed.sum()),
                      "mean_abs_input_change": float(np.abs(updated - original).sum() / mask.sum())}


def encode_input(model_name: str, model, volume: np.ndarray, device: torch.device) -> np.ndarray:
    batch = torch.from_numpy(volume[None, None]).to(device)
    with torch.inference_mode():
        if model_name == "tangerine":
            embedding = model.fc_norm(model.forward_features(batch))
        elif model_name == "radjepa":
            embedding = model(batch, indices=None).mean(dim=1)
        elif model_name == "ctclip":
            tokens = model["encoder"](batch, return_encoded_tokens=True)
            spatial = tokens.mean(dim=1).reshape(batch.shape[0], -1)
            embedding = torch.nn.functional.normalize(model["projection"](spatial), dim=-1)
        else:
            embedding = model.encode_image(batch, pool=True, project=True, normalize=False)
    vector = embedding.detach().float().cpu().numpy().reshape(-1)
    if vector.shape != (DIMENSIONS[model_name],) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid {model_name} global embedding")
    del batch, embedding
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fill-method", choices=("median", "inpaint", "black"), default="median")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Operational shard start in sorted patient order")
    parser.add_argument("--end-index", type=int, default=0,
                        help="Operational shard end (exclusive); zero means all")
    args = parser.parse_args()
    if args.max_patients < 0:
        parser.error("max-patients must be nonnegative")
    if args.start_index < 0 or args.end_index < 0:
        parser.error("shard indices must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    control_dir = OUTPUT / "control_masks"
    if not (control_dir / "COMPLETE").exists():
        raise ValueError("Prepare all same-shape control masks first")
    control = pd.read_csv(control_dir / "control_manifest.csv", dtype={"patient_id": str, "ct_series_uid": str})
    if control.patient_id.tolist() != data.patient_id.tolist():
        raise ValueError("Control mask manifest differs from cohort")
    input_dir = INPUTS / args.model / "original_ct"
    input_manifest = pd.read_csv(input_dir / "input_manifest.csv",
                                 dtype={"patient_id": str, "ct_series_uid": str})
    if input_manifest.patient_id.tolist() != data.patient_id.tolist():
        raise ValueError("Input manifest differs from cohort")
    output_folder = {"median": "embeddings", "inpaint": "embeddings_inpaint",
                     "black": "embeddings_black"}[args.fill_method]
    output = args.output_root / output_folder / args.model
    output.mkdir(parents=True, exist_ok=True)
    conditions = {"median": CONDITIONS,
                  "inpaint": ("gtv_inpaint", "sham_inpaint"),
                  "black": ("gtv_black", "sham_black")}[args.fill_method]
    protocol = {"model": args.model, "conditions": list(conditions),
                "checkpoint_sha256": sha256(CHECKPOINTS[args.model]),
                "cohort_sha256": sha256(COHORT),
                "input_manifest_sha256": sha256(input_dir / "input_manifest.csv"),
                "control_manifest_sha256": sha256(control_dir / "control_manifest.csv"),
                "fill": {"median": "median model-input intensity in local 5–20 mm exterior shell; full GTV core and 3 mm feather outside",
                         "inpaint": "nearest exterior intensity extrapolation, Gaussian smoothed at 2 mm; full GTV core and 3 mm feather outside",
                         "black": "set mask voxels only to model-input minimum intensity; no exterior feather"}[args.fill_method],
                **({"black_value": BLACK_VALUES[args.model]} if args.fill_method == "black" else {}),
                "encoder": "frozen official E1 global representation",
                "dtype": "cached float16 input cast to float32 before frozen encoder",
                "max_patients": args.max_patients}
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Masked embedding protocol differs")
    else:
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(protocol, indent=2) + "\n")
        temporary.replace(path)
    rows = data.to_dict("records")[:args.max_patients or None]
    shard = rows[args.start_index:args.end_index or None]
    control = control.set_index("patient_id")
    missing = [row for row in shard if not (output / "patients" / f"{row['patient_id']}.json").exists()]
    if missing:
        model, processor, _ = load_encoder(args.model, device, CHECKPOINTS[args.model])
        if args.model == "colipri":
            geometry.COLIPRI_PROCESSOR = processor
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        for i, row in enumerate(missing, 1):
            pid = row["patient_id"]
            volume = np.load(input_dir / f"{pid}.npy", allow_pickle=False).astype(np.float32)
            metadata = json.loads((geometry.MASKS / pid / f"{pid}_metadata.json").read_text())
            source = geometry.source_reference(metadata)
            mask = sitk.ReadImage(str(geometry.MASKS / pid / f"{pid}_gtv_mask.nii.gz"), sitk.sitkUInt8)
            native = sitk.Resample(mask, source, sitk.Transform(), sitk.sitkNearestNeighbor,
                                   0, sitk.sitkUInt8)
            gtv_array = sitk.GetArrayFromImage(native) > 0
            if not gtv_array.any():
                raise ValueError(f"{pid}: empty original CT GTV")
            entry = control.loc[pid]
            sham_stats = None
            sham_array = sham_native = sham_filled = None
            label_array = gtv_array.astype(np.uint8)
            if not pd.isna(entry.sham_dx_voxels):
                sham_array = shifted_mask(gtv_array, int(entry.sham_dx_voxels),
                                          int(entry.sham_dy_voxels))
                label_array += 2 * sham_array
            labels_native = sitk.GetImageFromArray(label_array)
            labels_native.CopyInformation(native)
            labels, spacing = to_input_labels(args.model, labels_native, volume.shape)
            gtv_input = labels == 1
            gtv_filled, gtv_stats = fill_mask(volume, gtv_input, spacing,
                                               args.fill_method, BLACK_VALUES[args.model])
            arrays = ({"original": volume, "gtv_filled": gtv_filled}
                      if args.fill_method == "median" else
                      {f"gtv_{args.fill_method}": gtv_filled})
            if sham_array is not None:
                sham_input = labels == 2
                sham_filled, sham_stats = fill_mask(volume, sham_input, spacing,
                                                    args.fill_method, BLACK_VALUES[args.model])
                arrays["sham_filled" if args.fill_method == "median"
                       else f"sham_{args.fill_method}"] = sham_filled
            for condition, array in arrays.items():
                vector = encode_input(args.model, model, array, device)
                p = output / condition / f"{pid}.npy"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, vector)
            record = {"patient_id": pid, "ct_series_uid": row["ct_series_uid"],
                      "gtv_input": gtv_stats, "sham_input": sham_stats}
            p = output / "patients" / f"{pid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(record, indent=2) + "\n")
            if i % 5 == 0 or i == len(missing):
                print(f"{args.model}: shard {args.start_index}:{args.end_index or len(rows)}, "
                      f"encoded {len(shard)-len(missing)+i}/{len(shard)}", flush=True)
            del volume, gtv_filled, sham_filled, native, sham_native, gtv_array, sham_array
    if len(rows) == len(data) and all(
        (output / "patients" / f"{pid}.json").exists() for pid in data.patient_id
    ):
        records = [json.loads((output / "patients" / f"{pid}.json").read_text())
                   for pid in data.patient_id]
        pd.json_normalize(records).to_csv(output / "input_change_manifest.csv", index=False)
        n_sham = sum(record["sham_input"] is not None for record in records)
        (output / "COMPLETE").write_text(
            f"{len(data)} patients GTV {args.fill_method}; {n_sham} matched sham {args.fill_method}\n")


if __name__ == "__main__":
    main()
