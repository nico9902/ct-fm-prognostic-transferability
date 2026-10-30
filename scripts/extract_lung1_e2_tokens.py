#!/usr/bin/env python3
"""Cache final frozen spatial tokens for the E2 SQA rung on original LUNG1 CTs.

Token grids larger than 8^3 are deterministically average pooled to 8^3.
The grid rule is fixed before survival fitting and independent of outcomes.
Requires a CUDA host; the CT-CLIP encoder itself assumes CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_lung1_e1_embeddings import (  # noqa: E402
    CHECKPOINTS, COHORT, ctclip_input, dinov3_forward, dinov3_input, load_encoder,
    radjepa_input, read_image, sha256, tangerine_input,
)


def encode_tokens(name: str, model, processor, image, device: torch.device):
    if name == "tangerine":
        batch = tangerine_input(image).to(device)
        with torch.inference_mode():
            patches = model.patch_embed(batch)
            x = torch.cat((model.cls_token.expand(len(batch), -1, -1), patches), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            for block in model.blocks:
                x = block(x)
            tokens = x[:, 1:]
            grid = tuple(int(v) for v in model.patch_embed.grid_size)
    elif name == "radjepa":
        batch = radjepa_input(image).to(device)
        with torch.inference_mode():
            tokens = model(batch, indices=None)
            metadata = json.loads((ROOT / "external/Rad-Jepa-3D-hf/config.json").read_text())
            grid = tuple(int(size // patch) for size, patch in
                         zip(metadata["input_shape"][-3:], metadata["patch_size"]))
            if getattr(model, "use_morton", False):
                inverse = torch.argsort(model._morton_order)
                tokens = tokens[:, inverse]
    elif name == "ctclip":
        batch = ctclip_input(image, device)
        with torch.inference_mode():
            native = model["encoder"](batch, return_encoded_tokens=True)
            grid = tuple(int(v) for v in native.shape[1:4])
            tokens = native.reshape(len(batch), -1, native.shape[-1])
    elif name == "dinov3":
        batch = dinov3_input(image).to(device)
        with torch.inference_mode():
            # Drop the class and register tokens; the slice axis is depth.
            sequence = dinov3_forward(model, batch)[:, 1 + model.config.num_register_tokens:].float()
            side = int(round(sequence.shape[1] ** 0.5))
            grid = (sequence.shape[0], side, side)
            tokens = sequence.reshape(1, -1, sequence.shape[-1])
    else:
        import torchio as tio
        batch = processor.to_images_batch(processor.process_images(tio.ScalarImage.from_sitk(image))).to(device)
        # COLIPRI builds a dense pooling module during this call, which requires
        # normal version counters even though gradients are disabled.
        with torch.no_grad():
            native = model.encode_image(batch, pool=False, project=True, normalize=False)
            grid = tuple(int(v) for v in native.shape[-3:])
            tokens = native.flatten(2).transpose(1, 2)
    if int(np.prod(grid)) != tokens.shape[1]:
        raise ValueError(f"{name}: token count {tokens.shape[1]} differs from grid {grid}")
    native_grid = grid
    if max(grid) > 8:
        tensor = tokens.transpose(1, 2).reshape(1, tokens.shape[-1], *grid)
        grid = tuple(min(v, 8) for v in grid)
        tokens = F.adaptive_avg_pool3d(tensor, grid).flatten(2).transpose(1, 2)
    array = tokens[0].detach().float().cpu().numpy()
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name}: invalid final token array")
    return array.astype(np.float16), native_grid, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patients", type=int, default=0, help="One-patient smoke use only")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/lung1_e2_tokens")
    args = parser.parse_args()
    if args.max_patients < 0:
        parser.error("max-patients must be nonnegative")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("E2 token extraction requires a visible CUDA GPU")
    checkpoint = CHECKPOINTS[args.model]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    if args.max_patients:
        data = data.head(args.max_patients)
    output = args.output_root / args.model / "original_ct"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": args.model, "source": "original_ct", "cohort_sha256": sha256(COHORT),
                "checkpoint_sha256": sha256(checkpoint), "token_layer": "final",
                "spatial_rule": "native_if_each_axis_le_8_else_adaptive_average_to_max_8_per_axis",
                "dtype": "float16", "max_patients": args.max_patients}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing token protocol differs; use a new output root")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    prior_manifest = output / "token_manifest.csv"
    prior = (pd.read_csv(prior_manifest, dtype=str).set_index("patient_id").to_dict("index")
             if prior_manifest.exists() else {})
    device = torch.device(args.device)
    model, processor, _ = load_encoder(args.model, device, checkpoint)
    records = []
    for index, row in data.iterrows():
        path = output / f"{row.patient_id}.npy"
        if path.exists():
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError(f"Invalid cached tokens: {path}")
            if row.patient_id not in prior:
                raise ValueError(f"Cached tokens lack manifest provenance: {path}")
            native_grid = json.loads(prior[row.patient_id]["native_grid"])
            grid = json.loads(prior[row.patient_id]["saved_grid"])
        else:
            image = read_image(row, "original_ct", ROOT / "data/processed/lung1/thorax_hu")
            array, native_grid, grid = encode_tokens(args.model, model, processor, image, device)
            temporary = output / f"{row.patient_id}.tmp.npy"
            np.save(temporary, array)
            temporary.replace(path)
        records.append({"patient_id": row.patient_id, "ct_series_uid": row.ct_series_uid,
                        "token_path": path.name, "n_tokens": array.shape[0], "dimension": array.shape[1],
                        "native_grid": json.dumps(native_grid), "saved_grid": json.dumps(grid)})
        temporary_manifest = output / "token_manifest.tmp.csv"
        with temporary_manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        temporary_manifest.replace(output / "token_manifest.csv")
        print(f"{args.model}: {index + 1}/{len(data)} {row.patient_id}", flush=True)


if __name__ == "__main__":
    main()
