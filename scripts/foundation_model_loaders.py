"""Shared pretrained encoder loaders for the CT prognostic benchmark."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


def add_radjepa_to_path(radjepa_code_dir: Path) -> None:
    sys.path.insert(0, str(radjepa_code_dir.resolve()))


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


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def build_tangerine_model(code, checkpoint, device):
    sys.path.insert(0, str(code.resolve()))
    import models_vit
    model = models_vit.vit_large_patch16_yo(img_size=256, num_classes=0, global_pool=True, drop_path_rate=0)
    # Official release is a trusted-source research checkpoint; inspect tensor mapping.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    md = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
    load = model.load_state_dict(filtered, strict=False)
    # Unlike official fine-tuning's permissive filter, never silently leave an encoder random.
    missing = set(load.missing_keys) - {"fc_norm.weight", "fc_norm.bias"}
    if missing:
        raise RuntimeError(f"Missing pretrained encoder weights: {sorted(missing)}")
    model.requires_grad_(False).eval().to(device)
    info = {"checkpoint_sha256": digest(checkpoint), "missing_keys": load.missing_keys,
            "ignored_keys": sorted(set(state) - set(filtered)), "loaded_keys": len(filtered),
            "tensor_shapes": {k: list(v.shape) for k, v in state.items() if torch.is_tensor(v)},
            "trainable_encoder_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, info

