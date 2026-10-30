#!/usr/bin/env python3
"""Render actual E1 input tensors for one LUNG1 patient, centered on the GTV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchio as tio

from extract_lung1_e1_embeddings import COHORT, ROOT, THORAX
from extract_lung1_e1_embeddings import ctclip_input, radjepa_input, read_image, tangerine_input


def gtv_center_lps(patient_id: str) -> tuple[float, float, float]:
    mask = sitk.ReadImage(str(THORAX / patient_id / f"{patient_id}_gtv_mask.nii.gz"))
    indices_zyx = np.argwhere(sitk.GetArrayFromImage(mask) > 0)
    if len(indices_zyx) == 0:
        raise ValueError(f"{patient_id}: empty GTV mask")
    index_xyz = tuple(float(value) for value in indices_zyx.mean(axis=0)[::-1])
    return mask.TransformContinuousIndexToPhysicalPoint(index_xyz)


def bounded_index(index_zyx, shape):
    return tuple(int(np.clip(round(value), 0, length - 1))
                 for value, length in zip(index_zyx, shape))


def make_examples(image: sitk.Image, center_lps, device: torch.device):
    examples = []

    tensor = tangerine_input(image)[0, 0].numpy()
    source_index_xyz = np.array(image.TransformPhysicalPointToContinuousIndex(center_lps))
    target_index_xyz = source_index_xyz * 256 / np.array(image.GetSize())
    spacing_xyz = tuple(n * spacing / 256 for n, spacing in zip(image.GetSize(), image.GetSpacing()))
    examples.append(("TANGERINE", tensor, bounded_index(target_index_xyz[::-1], tensor.shape),
                     spacing_xyz, (0, 1), "native orientation; B-spline; HU [−1200, 800]"))

    canonical = sitk.DICOMOrient(image, "RAS")
    tensor = radjepa_input(image)[0, 0].numpy()
    source_index_xyz = np.array(canonical.TransformPhysicalPointToContinuousIndex(center_lps))
    source_shape_zyx = np.array(canonical.GetSize()[::-1])
    target_index_zyx = source_index_xyz[::-1] * np.array(tensor.shape) / source_shape_zyx
    spacing_xyz = tuple(n * spacing / target for n, spacing, target in
                        zip(canonical.GetSize(), canonical.GetSpacing(), (256, 256, 32)))
    examples.append(("Rad-JEPA-3D", tensor, bounded_index(target_index_zyx, tensor.shape),
                     spacing_xyz, (0, 1), "RAS; linear resize; HU [−1000, 400]"))

    from colipri import get_processor
    processed = get_processor(image_only=True).process_images(tio.ScalarImage.from_sitk(image))[0]
    # TorchIO stores (C,X,Y,Z). Convert to the D,H,W convention used for display.
    tensor = processed.data[0].numpy().transpose(2, 1, 0)
    center_ras = np.array((-center_lps[0], -center_lps[1], center_lps[2], 1.0))
    target_index_xyz = np.linalg.solve(processed.affine, center_ras)[:3]
    examples.append(("COLIPRI", tensor, bounded_index(target_index_xyz[::-1], tensor.shape),
                     (2.0, 2.0, 2.0), (-1, 1), "RAS; 2 mm; center crop/pad; HU [−1000, 1000]"))

    tensor = ctclip_input(image, device)[0, 0].cpu().numpy()
    canonical = sitk.DICOMOrient(image, "LPS")
    source_index_xyz = np.array(canonical.TransformPhysicalPointToContinuousIndex(center_lps))
    target_spacing_xyz = np.array((0.75, 0.75, 1.5))
    resampled_size_xyz = np.array(tuple(max(1, int(n * old / new)) for n, old, new in
                                         zip(canonical.GetSize(), canonical.GetSpacing(), target_spacing_xyz)))
    resampled_index_xyz = source_index_xyz * np.array(canonical.GetSpacing()) / target_spacing_xyz
    target_size_xyz = np.array((480, 480, 240))
    crop_start_xyz = np.maximum((resampled_size_xyz - target_size_xyz) // 2, 0)
    pad_before_xyz = np.maximum(target_size_xyz - resampled_size_xyz, 0) // 2
    target_index_xyz = resampled_index_xyz - crop_start_xyz + pad_before_xyz
    examples.append(("CT-CLIP", tensor, bounded_index(target_index_xyz[::-1], tensor.shape),
                     tuple(target_spacing_xyz), (-1, 1),
                     "LPS; 0.75 × 0.75 × 1.5 mm; center crop/pad; HU [−1000, 1000]"))
    return examples


def draw(examples, patient_id: str, source: str, output: Path):
    count = len(examples)
    fig, axes = plt.subplots(count, 3, figsize=(14, 4 * count), constrained_layout=True)
    if count == 1:
        axes = np.asarray([axes])
    for row, (name, volume, (depth, height, width), spacing_xyz, value_range, description) in enumerate(examples):
        spacing_x, spacing_y, spacing_z = spacing_xyz
        views = (
            (volume[depth], "Axial", spacing_x, spacing_y),
            (volume[:, height, :], "Coronal", spacing_x, spacing_z),
            (volume[:, :, width], "Sagittal", spacing_y, spacing_z),
        )
        for col, (plane, title, horizontal_mm, vertical_mm) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(plane, cmap="gray", vmin=value_range[0], vmax=value_range[1],
                      interpolation="nearest", origin="lower",
                      extent=(0, plane.shape[1] * horizontal_mm, 0, plane.shape[0] * vertical_mm))
            ax.set_aspect("equal")
            ax.set_title(f"{title} • {plane.shape[1]} × {plane.shape[0]} pixels", fontsize=11)
            ax.set_xlabel("mm")
            ax.set_ylabel("mm")
        axes[row, 0].text(0, 1.12, f"{name} — input {volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]} (D × H × W)\n{description}; normalized {value_range}",
                          transform=axes[row, 0].transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.suptitle(f"{patient_id}: {source} CT inputs before frozen encoders, slices through GTV centroid\n"
                 "The GTV selects slice positions for display only; no mask or overlay entered any encoder.",
                 fontsize=14)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", default="LUNG1-001")
    parser.add_argument("--source", choices=("original_ct", "thorax_hu"), default="original_ct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    selected = data.loc[data.patient_id.eq(args.patient_id) & data.region_status.eq("ready")]
    if len(selected) != 1:
        raise ValueError(f"Expected one eligible cohort row for {args.patient_id}")
    image = read_image(next(selected.itertuples(index=False)), args.source, THORAX)
    source_values = sitk.GetArrayFromImage(image)
    source_min_max = (float(source_values.min()), float(source_values.max()))
    del source_values
    center = gtv_center_lps(args.patient_id)
    examples = make_examples(image, center, torch.device(args.device))
    output_dir = args.output_dir or ROOT / "outputs/lung1_e1_input_examples" / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = output_dir / f"{args.patient_id}_comparison.png"
    draw(examples, args.patient_id, args.source, comparison)
    print(comparison)
    audit = {"patient_id": args.patient_id, "source": args.source,
             "source_size_xyz": image.GetSize(), "source_spacing_xyz_mm": image.GetSpacing(),
             "source_hu_min_max": source_min_max, "gtv_centroid_lps_mm": center, "encoders": {}}
    for example in examples:
        name = example[0].lower().replace("-", "")
        path = output_dir / f"{args.patient_id}_{name}.png"
        draw([example], args.patient_id, args.source, path)
        print(path)
        audit["encoders"][example[0]] = {
            "tensor_shape_dhw": example[1].shape,
            "selected_slice_dhw": example[2],
            "normalized_min_max": (float(example[1].min()), float(example[1].max())),
            "figure": str(path),
        }
        print(example[0], "tensor_shape_DHW", example[1].shape,
              "selected_slice_DHW", example[2], "value_min_max",
              (float(example[1].min()), float(example[1].max())))
    audit_path = output_dir / f"{args.patient_id}_input_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(audit_path)


if __name__ == "__main__":
    main()
