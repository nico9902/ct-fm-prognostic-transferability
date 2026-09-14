#!/usr/bin/env python3
"""
Extract frozen Rad-JEPA-3D embeddings from prepared .npy CT volumes.

This script requires the original RadJepa code and mamba-ssm. On macOS CPU/MPS,
mamba-ssm may not install; run this step in a CUDA/Linux environment if needed.
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


def add_radjepa_to_path(radjepa_code_dir: Path) -> None:
    sys.path.insert(0, str(radjepa_code_dir.resolve()))


def device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_encoder(radjepa_code_dir: Path, config_path: Path, checkpoint_path: Path, device: torch.device):
    add_radjepa_to_path(radjepa_code_dir)
    try:
        from vjepa_config import VJEPAConfig
        from vjepa_model import MambaEncoder
    except ModuleNotFoundError as e:
        if e.name == "mamba_ssm":
            raise SystemExit(
                "Missing dependency 'mamba_ssm'. Rad-JEPA-3D's encoder uses Mamba blocks.\n"
                "Install/run this step in a CUDA/Linux environment, for example:\n"
                "  pip install mamba-ssm causal-conv1d\n"
            ) from e
        raise

    with config_path.open() as f:
        hf_cfg = json.load(f)

    cfg = VJEPAConfig()
    cfg.dim = int(hf_cfg.get("hidden_size", cfg.dim))
    cfg.n_layers = int(hf_cfg.get("num_hidden_layers", cfg.n_layers))
    cfg.patch_size = tuple(hf_cfg.get("patch_size", cfg.patch_size))
    cfg.volume_size = tuple(hf_cfg.get("input_shape", [1, *cfg.volume_size])[1:])
    cfg.use_hybrid = True
    cfg.router_mode = hf_cfg.get("router_mode", "layer")
    cfg.use_morton_scan = hf_cfg.get("scan_order", "raster") == "morton"
    cfg.use_rope = hf_cfg.get("position_encoding", "rope_3d") == "rope_3d"

    encoder = MambaEncoder(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("target_encoder") or checkpoint.get("context_encoder") or checkpoint
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, cfg


def load_batch(paths: list[str], device: torch.device) -> torch.Tensor:
    arrays = [np.load(path).astype(np.float32, copy=False) for path in paths]
    batch = torch.from_numpy(np.stack(arrays, axis=0))
    return batch.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("data/processed/nlst/radjepa_npy_manifest.csv"),
    )
    parser.add_argument("--radjepa-code-dir", type=Path, default=Path("external/RadJepa"))
    parser.add_argument("--hf-model-dir", type=Path, default=Path("external/Rad-Jepa-3D-hf"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/embeddings/radjepa3d"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("outputs/embeddings/radjepa3d_manifest.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-mode",
        choices=["pooled", "tokens"],
        default="pooled",
        help="Save mean-pooled embeddings or the full Rad-JEPA token sequence.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or args.hf_model_dir / "encoder" / "encoder.pt"
    config = args.config or args.hf_model_dir / "config.json"
    device = device_from_arg(args.device)
    print(f"[radjepa] device={device}")

    encoder, cfg = build_encoder(args.radjepa_code_dir, config, checkpoint, device)
    manifest = pd.read_csv(args.input_manifest, dtype=str)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pending = []
    for _, row in manifest.iterrows():
        patient_id = str(row["patient_id"])
        timepoint = str(row["timepoint"])
        series_uid = str(row["series_uid"])
        out_dir = args.output_root / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{patient_id}_{timepoint}_{series_uid}.npy"
        record = row.to_dict()
        record["embedding_path"] = str(out_path)

        if out_path.exists() and not args.overwrite:
            existing = np.load(out_path, mmap_mode="r")
            record["embedding_dim"] = str(existing.shape[-1])
            record["num_tokens"] = str(existing.shape[0]) if existing.ndim == 2 else "1"
            record["embedding_output_mode"] = args.output_mode
            record["embedding_status"] = "exists"
            rows.append(record)
            continue
        pending.append((record, row["radjepa_npy_path"], out_path))

    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            paths = [item[1] for item in chunk]
            batch = load_batch(paths, device)
            tokens = encoder(batch, indices=None)
            if args.output_mode == "tokens":
                embeddings = tokens.detach().cpu().numpy().astype(np.float32)
            else:
                embeddings = tokens.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            for emb, (record, _, out_path) in zip(embeddings, chunk):
                np.save(out_path, emb)
                record["embedding_dim"] = str(emb.shape[-1])
                record["num_tokens"] = str(emb.shape[0]) if emb.ndim == 2 else "1"
                record["embedding_output_mode"] = args.output_mode
                record["embedding_status"] = "extracted"
                rows.append(record)
                print(f"[{len(rows):03d}/{len(manifest):03d}] extracted: {out_path}")

    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. manifest={args.output_manifest} dim={cfg.dim}")


if __name__ == "__main__":
    main()
