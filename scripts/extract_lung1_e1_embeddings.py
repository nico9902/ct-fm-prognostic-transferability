#!/usr/bin/env python3
"""Extract frozen global LUNG1 embeddings for the E1 Cox experiment.

The original_ct source is the proposal's model-specific primary input. The
thorax_hu source is the GTV-preserving common-crop sensitivity input. Both are
subsequently transformed with each encoder's prescribed input processing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
THORAX = ROOT / "data/processed/lung1/thorax_hu"
OUTPUT = ROOT / "outputs/lung1_e1_embeddings"
CHECKPOINTS = {
    "tangerine": ROOT / "external/tangerine/tangerine.pth",
    "radjepa": ROOT / "external/Rad-Jepa-3D-hf/encoder/encoder.pt",
    "colipri": ROOT / "external/colipri/model.safetensors",
    "ctclip": ROOT / "external/ctclip/CT-CLIP_v2.pt",
    "dinov3": ROOT / "external/dinov3-vitb16-pretrain-lvd1689m/model.safetensors",
}
DIMENSIONS = {"tangerine": 2048, "radjepa": 384, "colipri": 768, "ctclip": 512, "dinov3": 768}
# DINOv3 is a 2-D encoder applied to axial slices; values follow its HF processor.
DINOV3_SIZE = 224
DINOV3_SLICE_MM = 3.0
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)
DINOV3_CHUNK = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(row, source: str, thorax_root: Path) -> sitk.Image:
    if source == "thorax_hu":
        folder = thorax_root / row.patient_id
        metadata = json.loads((folder / f"{row.patient_id}_metadata.json").read_text())
        if metadata.get("mode") != "thorax_hu" or metadata["source_ct_series_uid"] != row.ct_series_uid:
            raise ValueError(f"{row.patient_id}: thorax crop provenance differs from cohort")
        if metadata["gtv_fraction_inside_crop"] != 1.0:
            raise ValueError(f"{row.patient_id}: GTV was not completely retained")
        image = sitk.ReadImage(str(folder / f"{row.patient_id}_volume.nii.gz"), sitk.sitkFloat32)
    else:
        paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(row.ct_series_dir))
        if len(paths) != int(row.ct_found_slices):
            raise ValueError(f"{row.patient_id}: DICOM slice count differs from cohort audit")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(paths)
        reader.SetOutputPixelType(sitk.sitkFloat32)
        image = reader.Execute()
    if image.GetDimension() != 3:
        raise ValueError(f"{row.patient_id}: expected a 3-D CT")
    return image


def tangerine_input(image: sitk.Image) -> torch.Tensor:
    resample = sitk.ResampleImageFilter()
    resample.SetSize((256, 256, 256))
    resample.SetOutputSpacing([n * spacing / 256 for n, spacing in zip(image.GetSize(), image.GetSpacing())])
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkBSpline)
    volume = sitk.GetArrayFromImage(resample.Execute(image)).astype(np.float32)
    volume = (np.clip(volume, -1200, 800) + 1200) / 2000
    return torch.from_numpy(volume[None, None])


def radjepa_input(image: sitk.Image) -> torch.Tensor:
    from scipy.ndimage import zoom
    # Rad-JEPA input convention: canonical RAS, [D=S, H=A, W=R],
    # [-1000,400] HU, then linear resizing to 32 x 256 x 256.
    canonical = sitk.DICOMOrient(image, "RAS")
    volume = sitk.GetArrayFromImage(canonical).astype(np.float32)
    volume = (np.clip(volume, -1000, 400) + 1000) / 1400
    scale = [target / current for target, current in zip((32, 256, 256), volume.shape)]
    volume = zoom(volume, zoom=scale, order=1).astype(np.float32, copy=False)
    if volume.shape != (32, 256, 256):
        raise ValueError(f"Rad-JEPA resize produced {volume.shape}")
    return torch.from_numpy(volume[None, None])


def ctclip_input(image: sitk.Image, device: torch.device) -> torch.Tensor:
    # CT-CLIP data_inference_nii.py: 0.75 x 0.75 x 1.5 mm, HU [-1000,1000],
    # center crop/pad to 480 x 480 x 240 (x,y,z), then output (C,D,H,W).
    image = sitk.DICOMOrient(image, "LPS")
    spacing_xyz = (0.75, 0.75, 1.5)
    size_xyz = tuple(max(1, int(n * old / new)) for n, old, new in
                     zip(image.GetSize(), image.GetSpacing(), spacing_xyz))
    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    batch = torch.from_numpy(volume[None, None]).to(device)
    batch = torch.nn.functional.interpolate(batch, size=size_xyz[::-1], mode="trilinear",
                                             align_corners=False)
    batch = batch.clamp_(-1000, 1000).div_(1000)
    # The author's tensor is (D,H,W) after center crop/pad of (X,Y,Z).
    shape = (240, 480, 480)
    cuts = []
    padding = []
    for length, target in zip(batch.shape[-3:], shape):
        start = max((length - target) // 2, 0)
        cuts.append(slice(start, min(start + target, length)))
        remaining = target - min(length, target)
        padding.append((remaining // 2, remaining - remaining // 2))
    batch = batch[:, :, cuts[0], cuts[1], cuts[2]]
    batch = torch.nn.functional.pad(batch, tuple(value for pair in padding[::-1] for value in pair),
                                    value=-1)
    if tuple(batch.shape) != (1, 1, *shape):
        raise ValueError(f"CT-CLIP resize produced {tuple(batch.shape)}")
    return batch


def dinov3_input(image: sitk.Image) -> torch.Tensor:
    # LPS array order gives the radiological axial view (anterior up, patient
    # right on image left). Slices are linearly resampled to 3 mm, HU [-1000,400]
    # is mapped to [0,1], then each slice is resized like the HF processor
    # (bilinear, antialiased) to 224 x 224. Output is (1, 1, slices, 224, 224).
    canonical = sitk.DICOMOrient(image, "LPS")
    volume = sitk.GetArrayFromImage(canonical).astype(np.float32)
    volume = (np.clip(volume, -1000, 400) + 1000) / 1400
    batch = torch.from_numpy(volume)[None, None]
    slices = max(1, int(round(volume.shape[0] * canonical.GetSpacing()[2] / DINOV3_SLICE_MM)))
    batch = torch.nn.functional.interpolate(batch, size=(slices, *volume.shape[1:]),
                                            mode="trilinear", align_corners=False)[0, 0]
    batch = torch.nn.functional.interpolate(batch[:, None], size=(DINOV3_SIZE, DINOV3_SIZE),
                                            mode="bilinear", align_corners=False, antialias=True)
    return batch[:, 0].clamp_(0, 1)[None, None]


def dinov3_forward(model, batch: torch.Tensor) -> torch.Tensor:
    """Final normed token sequences (slices, 1 + registers + patches, 768) of a (1,1,Z,H,W) volume."""
    slices = batch[0, 0][:, None].expand(-1, 3, -1, -1)
    mean = torch.tensor(DINOV3_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(DINOV3_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    slices = (slices - mean) / std
    return torch.cat([model(pixel_values=slices[start:start + DINOV3_CHUNK]).last_hidden_state
                      for start in range(0, len(slices), DINOV3_CHUNK)])


def load_encoder(model_name: str, device: torch.device, checkpoint: Path):
    if model_name == "dinov3":
        from transformers import DINOv3ViTModel
        model = DINOv3ViTModel.from_pretrained(checkpoint.parent, attn_implementation="sdpa")
        model.requires_grad_(False).eval().to(device)
        return model, None, {"representation": "slice_mean_final_norm_cls",
                             "num_register_tokens": model.config.num_register_tokens}
    if model_name == "tangerine":
        sys.path.insert(0, str(ROOT / "scripts"))
        from foundation_model_loaders import build_tangerine_model as build_model
        model, details = build_model(ROOT / "external/tangerine", checkpoint, device)
        return model, None, {"representation": "pre_logits_fc_norm_cls_plus_mean_patch", **details}
    if model_name == "radjepa":
        sys.path.insert(0, str(ROOT / "scripts"))
        from foundation_model_loaders import build_encoder
        model, config = build_encoder(ROOT / "external/RadJepa",
                                      ROOT / "external/Rad-Jepa-3D-hf/config.json", checkpoint, device)
        return model, None, {"representation": "mean_final_tokens", "hidden_size": config.dim}
    if model_name == "ctclip":
        sys.path.insert(0, str(ROOT / "third_party/ctclip"))
        from transformer_maskgit import CTViT
        image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480,
                              patch_size=20, temporal_patch_size=10,
                              spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        image_state = {key.removeprefix("visual_transformer."): value for key, value in state.items()
                       if key.startswith("visual_transformer.")}
        image_encoder.load_state_dict(image_state, strict=True)
        projection = torch.nn.Linear(294912, 512, bias=False)
        projection.load_state_dict({"weight": state["to_visual_latent.weight"]}, strict=True)
        del state, image_state
        model = torch.nn.ModuleDict({"encoder": image_encoder, "projection": projection})
        model.requires_grad_(False).eval().to(device)
        return model, None, {"representation": "official_temporal_mean_spatial_flatten_projected_l2",
                             "architecture_source_commit": "a2a155c601987820433c01db69b64d701d3d229d"}
    sys.path.insert(0, str(ROOT / "external/colipri/deps"))
    from colipri import get_model, get_processor
    from safetensors import safe_open
    from safetensors.torch import load_file
    model = get_model(pretrained=False, image_only=True)
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        for key, value in model.state_dict().items():
            if key not in stream.keys() or tuple(stream.get_slice(key).get_shape()) != tuple(value.shape):
                raise ValueError(f"COLIPRI checkpoint missing or mismatched tensor: {key}")
    state = load_file(str(checkpoint))
    model.load_state_dict({key: state[key] for key in model.state_dict()}, strict=True)
    del state
    model.requires_grad_(False).eval().to(device)
    return model, get_processor(image_only=True), {"representation": "official_projected_attention_pool"}


def encode(model_name: str, model, processor, image: sitk.Image, device: torch.device) -> np.ndarray:
    if model_name == "tangerine":
        batch = tangerine_input(image).to(device)
        with torch.inference_mode():
            embedding = model.fc_norm(model.forward_features(batch))
    elif model_name == "radjepa":
        batch = radjepa_input(image).to(device)
        with torch.inference_mode():
            embedding = model(batch, indices=None).mean(dim=1)
    elif model_name == "ctclip":
        batch = ctclip_input(image, device)
        with torch.inference_mode():
            tokens = model["encoder"](batch, return_encoded_tokens=True)
            spatial = tokens.mean(dim=1).reshape(batch.shape[0], -1)
            embedding = torch.nn.functional.normalize(model["projection"](spatial), dim=-1)
    elif model_name == "dinov3":
        batch = dinov3_input(image).to(device)
        with torch.inference_mode():
            embedding = dinov3_forward(model, batch)[:, 0].float().mean(dim=0)
    else:
        import torchio as tio
        batch = processor.to_images_batch(processor.process_images(tio.ScalarImage.from_sitk(image))).to(device)
        with torch.inference_mode():
            embedding = model.encode_image(batch, pool=True, project=True, normalize=False)
    vector = embedding.detach().float().cpu().numpy().reshape(-1)
    if vector.shape != (DIMENSIONS[model_name],) or not np.isfinite(vector).all():
        raise ValueError(f"{model_name}: invalid embedding {vector.shape}")
    return vector


def write_manifest(records, path):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("patient_id", "ct_series_uid", "embedding_path"))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda record: record["patient_id"]))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--source", choices=("original_ct", "thorax_hu"), default="original_ct")
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--thorax-root", type=Path, default=THORAX)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patients", type=int, help="Smoke-test subset; cannot be used in the final E1 run")
    parser.add_argument("--status-column", default="region_status",
                        help="Cohort column whose 'ready' rows are encoded (Radiogenomics: global_status)")
    parser.add_argument("--clamp-min-hu", type=float,
                        help="Set voxels below this HU to it at read time (Radiogenomics padding: -1024)")
    args = parser.parse_args()
    if args.max_patients is not None and args.max_patients < 1:
        parser.error("--max-patients must be positive")
    checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data = pd.read_csv(args.cohort, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data[args.status_column].eq("ready")].sort_values("patient_id").reset_index(drop=True)
    if data.patient_id.duplicated().any() or data.empty:
        raise ValueError("Eligible cohort must have unique patients")
    if args.max_patients:
        data = data.head(args.max_patients)
    output = args.output_root / args.model / args.source
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "embedding_manifest.csv"
    provenance = output / "provenance.json"
    expected = {"model": args.model, "source": args.source, "checkpoint_sha256": sha256(checkpoint),
                "cohort_sha256": sha256(args.cohort), "dimension": DIMENSIONS[args.model],
                "preprocessing": {"tangerine": "bspline_256cubed_HU_-1200_800",
                                  "radjepa": "RAS_linear_32x256x256_HU_-1000_400",
                                  "colipri": "official_processor_RAS_2mm_192cubed",
                                  "ctclip": "LPS_trilinear_0.75x0.75x1.5mm_center480x480x240_HU_-1000_1000",
                                  "dinov3": "LPS_axial_3mm_slices_224x224_HU_-1000_400_gray_x3_imagenet_norm"}[args.model]}
    if args.clamp_min_hu is not None:
        expected["clamp_min_hu"] = args.clamp_min_hu
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"Cached embeddings use different {key}; select a new output directory")
    else:
        provenance.write_text(json.dumps(expected, indent=2) + "\n")
    device = torch.device(args.device)
    model, processor, details = load_encoder(args.model, device, checkpoint)
    if not provenance.exists():
        raise RuntimeError("Failed to write provenance")
    expected.update(details)
    provenance.write_text(json.dumps(expected, indent=2, default=str) + "\n")
    records = []
    for index, row in data.iterrows():
        path = output / f"{row.patient_id}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False)
            if vector.shape != (DIMENSIONS[args.model],) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid cached embedding: {path}")
        else:
            image = read_image(row, args.source, args.thorax_root)
            if args.clamp_min_hu is not None:
                image = sitk.Maximum(image, float(args.clamp_min_hu))
            vector = encode(args.model, model, processor, image, device)
            temporary = output / f"{row.patient_id}.tmp.npy"
            np.save(temporary, vector)
            temporary.replace(path)
        records.append({"patient_id": row.patient_id, "ct_series_uid": row.ct_series_uid,
                        "embedding_path": str(path.resolve())})
        write_manifest(records, manifest)
        print(f"{args.model}/{args.source}: {index + 1}/{len(data)} {row.patient_id}", flush=True)
    print(f"Complete: {manifest}")


if __name__ == "__main__":
    main()
