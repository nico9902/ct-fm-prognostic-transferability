#!/usr/bin/env python3
"""E4 token masking: frozen encoders with GTV (or control) tokens hidden inside the encoder.

The CT input is unchanged. For a masked condition, the native tokens that touch
the GTV (or the count-matched control) are hidden from every other token in
every layer, and their own outputs are excluded from every readout:

* TANGERINE, COLIPRI, DINOv3: their keys are masked in every self-attention.
* CT-CLIP: keys masked in every spatial and temporal attention, and the tokens
  are zeroed at the input of every PEG depthwise convolution (the only other
  path that mixes tokens).
* Rad-JEPA-3D: Mamba blocks have no attention to mask, so the tokens are removed
  from the sequence with the encoder's own JEPA context ``indices`` argument.

For a pure transformer, masking keys and excluding outputs is the same as
dropping tokens while keeping their positional encodings. Per patient we save
token-mean vectors over identical retained positions in every condition, plus
each model's official global readout computed over the retained tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, pack

from extract_lung1_e1_embeddings import (
    CHECKPOINTS, COHORT, DIMENSIONS, DINOV3_MEAN, DINOV3_STD, ROOT, load_encoder, sha256,
)


INPUTS = ROOT / "outputs/lung1_e2_inputs"
GEOMETRY = ROOT / "outputs/lung1_e4_token_masking/geometry"
OUTPUT = ROOT / "outputs/lung1_e4_token_masking/embeddings"
CONDITIONS = ("original", "gtv_masked", "control_masked")
DINOV3_CHUNK = 32


def timm_block(block, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """timm 0.9 ViT Block with a key mask (timm 0.9 Attention takes no mask)."""
    attn = block.attn
    y = block.norm1(x)
    B, N, C = y.shape
    qkv = attn.qkv(y).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
    y = attn.proj_drop(attn.proj(y.transpose(1, 2).reshape(B, N, C)))
    x = x + block.drop_path1(block.ls1(y))
    return x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))


def forward_tangerine(model, batch, keep):
    x = model.patch_embed(batch)
    x = torch.cat((model.cls_token.expand(len(x), -1, -1), x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    full = torch.cat((torch.ones(1, dtype=torch.bool, device=x.device), keep))[None]
    for block in model.blocks:
        x = timm_block(block, x, full)
    tokens = x[0, 1:]
    pooled = tokens[keep].mean(0)
    global_vector = model.fc_norm(torch.cat((x[0, 0], pooled))[None])[0]
    return tokens, global_vector


def forward_radjepa(model, batch, keep):
    if getattr(model, "use_morton", False):
        raise ValueError("Rad-JEPA token order is raster in this checkpoint")
    indices = torch.nonzero(keep).reshape(1, -1)
    kept = model(batch, indices=None if bool(keep.all()) else indices)[0]
    tokens = torch.zeros(len(keep), kept.shape[-1], device=kept.device, dtype=kept.dtype)
    tokens[keep] = kept
    return tokens, kept.mean(0)


def ctclip_transformer(transformer, x, video_shape, attn_bias, keep_rows):
    flat = keep_rows.to(x.dtype)[..., None]
    for peg, self_attn, cross_attn, ff in transformer.layers:
        if peg is not None:
            # Hidden tokens enter the depthwise PEG convolution as zeros (like padding).
            x = peg(x * flat, shape=video_shape) + x
        x = self_attn(x, attn_bias=attn_bias, mask=keep_rows) + x
        x = ff(x) + x
    return transformer.norm_out(x)


def forward_ctclip(model, batch, keep):
    encoder = model["encoder"]
    tokens = encoder.to_patch_emb(batch)
    b, t, h, w, d = tokens.shape
    video_shape = (b, t, h, w)
    keep = keep.reshape(t, h, w)
    x = rearrange(tokens, "b t h w d -> (b t) (h w) d")
    bias = encoder.spatial_rel_pos_bias(h, w, device=x.device)
    x = ctclip_transformer(encoder.enc_spatial_transformer, x, video_shape, bias,
                           keep.reshape(t, h * w))
    x = rearrange(x, "(b t) (h w) d -> (b h w) t d", b=b, h=h, w=w)
    x = ctclip_transformer(encoder.enc_temporal_transformer, x, video_shape, None,
                           keep.permute(1, 2, 0).reshape(h * w, t))
    x = rearrange(x, "(b h w) t d -> b t h w d", b=b, h=h, w=w)
    x, _ = pack([x], "b * d")
    x, _, _ = encoder.vq(x)
    x = rearrange(x, "b (t h w) d -> b t h w d", h=h, w=w)[0]
    weights = keep.to(x.dtype)[..., None]
    counts = weights.sum(0)
    spatial = (x * weights).sum(0) / counts.clamp_min(1)
    if bool((counts == 0).any()):
        raise ValueError("A CT-CLIP spatial position is hidden at every temporal token")
    global_vector = F.normalize(model["projection"](spatial.reshape(1, -1)), dim=-1)[0]
    return x.reshape(-1, d), global_vector


def forward_dinov3(model, batch, keep):
    slices = batch[0, 0][:, None].expand(-1, 3, -1, -1)
    mean = torch.tensor(DINOV3_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(DINOV3_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    slices = (slices - mean) / std
    prefix = 1 + model.config.num_register_tokens
    keep = keep.reshape(len(slices), -1)
    outputs = []
    for start in range(0, len(slices), DINOV3_CHUNK):
        pixels = slices[start:start + DINOV3_CHUNK].to(model.embeddings.patch_embeddings.weight.dtype)
        hidden = model.embeddings(pixels)
        rope = model.rope_embeddings(pixels)
        visible = torch.cat((torch.ones(len(pixels), prefix, dtype=torch.bool, device=hidden.device),
                             keep[start:start + DINOV3_CHUNK]), dim=1)[:, None, None, :]
        for layer in model.layer:
            hidden = layer(hidden, attention_mask=visible, position_embeddings=rope)
        outputs.append(model.norm(hidden))
    sequence = torch.cat(outputs).float()
    return sequence[:, prefix:].reshape(-1, sequence.shape[-1]), sequence[:, 0].mean(0)


def forward_colipri(model, batch, keep):
    image = model.image_encoder
    backbone = image.backbone
    x = backbone.down_projection(batch)
    B, C, W, H, D = x.shape
    x = rearrange(x, "b c w h d -> b (w h d) c")
    n_registers = 0
    if backbone.register_tokens is not None:
        n_registers = backbone.register_tokens.shape[1]
        x = torch.cat((backbone.register_tokens.expand(B, -1, -1), x), dim=1)
    x, rope, keep_indices = backbone.eva._pos_embed(x)
    if keep_indices is not None:
        raise ValueError("COLIPRI patch dropout is active")
    visible = torch.cat((torch.ones(n_registers, dtype=torch.bool, device=x.device), keep))
    visible = visible[None, None, None, :]
    for block in backbone.eva.blocks:
        if not block.attn.fused_attn:
            raise ValueError("timm Eva non-fused attention mishandles attn_mask")
        x = block(x, rope=rope, attn_mask=visible)
    x = backbone.eva.norm(x)[:, n_registers:]
    x = backbone.up_projection(rearrange(x, "b (w h d) c -> b c w h d", w=W, h=H, d=D))
    embeddings = image.projector(x)
    tokens = image.pooler.to_dense()(embeddings).flatten(2)[0].transpose(0, 1)
    sequence = rearrange(embeddings, "b c x y z -> b (x y z) c")
    global_vector = image.pooler(sequence, attention_mask=keep[None])[0]
    return tokens, global_vector


FORWARD = {"tangerine": forward_tangerine, "radjepa": forward_radjepa, "ctclip": forward_ctclip,
           "dinov3": forward_dinov3, "colipri": forward_colipri}


def official(model_name: str, model, batch):
    """Unmodified E1 global and E2 native-token forward, for exactness checks."""
    if model_name == "tangerine":
        x = model.patch_embed(batch)
        x = torch.cat((model.cls_token.expand(len(x), -1, -1), x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        for block in model.blocks:
            x = block(x)
        return x[0, 1:], model.fc_norm(model.forward_features(batch))[0]
    if model_name == "radjepa":
        tokens = model(batch, indices=None)[0]
        return tokens, tokens.mean(0)
    if model_name == "ctclip":
        tokens = model["encoder"](batch, return_encoded_tokens=True)
        spatial = tokens.mean(dim=1).reshape(1, -1)
        return tokens[0].reshape(-1, tokens.shape[-1]), F.normalize(model["projection"](spatial), dim=-1)[0]
    if model_name == "dinov3":
        from extract_lung1_e1_embeddings import dinov3_forward
        sequence = dinov3_forward(model, batch).float()
        prefix = 1 + model.config.num_register_tokens
        return sequence[:, prefix:].reshape(-1, sequence.shape[-1]), sequence[:, 0].mean(0)
    native = model.encode_image(batch, pool=False, project=True, normalize=False)
    pooled = model.encode_image(batch, pool=True, project=True, normalize=False)
    return native.flatten(2)[0].transpose(0, 1), pooled[0]


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(FORWARD), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--verify", type=int, default=3,
                        help="Check the unmasked forward against the official forward on this many new patients")
    args = parser.parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    if not (GEOMETRY / "COMPLETE").exists():
        raise ValueError("Run audit_lung1_e4_native_token_mapping.py first")
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    input_dir = INPUTS / args.model / "original_ct"
    manifest = pd.read_csv(input_dir / "input_manifest.csv", dtype={"patient_id": str})
    if manifest.patient_id.tolist() != data.patient_id.tolist():
        raise ValueError("Input manifest differs from cohort")
    output = OUTPUT / args.model
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": args.model, "conditions": list(CONDITIONS),
                "checkpoint_sha256": sha256(CHECKPOINTS[args.model]),
                "cohort_sha256": sha256(COHORT),
                "input_manifest_sha256": sha256(input_dir / "input_manifest.csv"),
                "geometry_protocol_sha256": sha256(GEOMETRY / "protocol.json"),
                "geometry_table_sha256": sha256(GEOMETRY / "patient_model_geometry.csv"),
                "mechanism": {"radjepa": "drop hidden tokens with encoder indices (JEPA context path)",
                              "ctclip": "key mask in spatial and temporal attention; hidden tokens zeroed at every PEG input"
                              }.get(args.model, "key mask in every self-attention layer"),
                "readouts": {"keep_gtv": "mean of final native tokens outside GTV tokens",
                             "keep_both": "mean of final native tokens outside GTV and control tokens",
                             "global": "official E1 global readout over retained tokens",
                             "all": "mean of all final native tokens (original only)"},
                "precision": "float32", "max_patients": args.max_patients}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Token-masking protocol differs")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(protocol, indent=2) + "\n")
    temporary.replace(path)
    ids = data.patient_id.tolist()[:args.max_patients or None]
    missing = [pid for pid in ids if not (output / "patients" / f"{pid}.json").exists()]
    if missing:
        model, _, _ = load_encoder(args.model, device, CHECKPOINTS[args.model])
        verified = 0
        for i, pid in enumerate(missing, 1):
            volume = np.load(input_dir / f"{pid}.npy", allow_pickle=False).astype(np.float32)
            batch = torch.from_numpy(volume[None, None]).to(device)
            masks = np.load(GEOMETRY / "masks" / args.model / f"{pid}.npz")
            gtv = torch.from_numpy(masks["gtv"].reshape(-1)).to(device)
            control = (torch.from_numpy(masks["control"].reshape(-1)).to(device)
                       if "control" in masks.files else None)
            hidden = {"original": torch.zeros_like(gtv), "gtv_masked": gtv}
            if control is not None:
                if bool((control & gtv).any()) or int(control.sum()) != int(gtv.sum()):
                    raise ValueError(f"{pid}: invalid control token mask")
                hidden["control_masked"] = control
            keep_gtv = ~gtv
            keep_both = ~(gtv | control) if control is not None else None
            vectors, record = {}, {"patient_id": pid, "n_tokens": int(len(gtv)),
                                   "n_gtv_tokens": int(gtv.sum()),
                                   "n_control_tokens": None if control is None else int(control.sum())}
            # no_grad, not inference_mode: COLIPRI builds its dense pooling module per call.
            with torch.no_grad():
                if verified < args.verify:
                    reference_tokens, reference_global = official(args.model, model, batch)
                outputs = {}
                for condition, mask in hidden.items():
                    tokens, global_vector = FORWARD[args.model](model, batch, ~mask)
                    tokens = tokens.float()
                    outputs[condition] = tokens
                    vectors[f"{condition}__global"] = global_vector.float()
                    if condition != "control_masked":
                        vectors[f"{condition}__keep_gtv"] = tokens[keep_gtv].mean(0)
                    if keep_both is not None:
                        vectors[f"{condition}__keep_both"] = tokens[keep_both].mean(0)
                vectors["original__all"] = outputs["original"].mean(0)
                if verified < args.verify:
                    record["verify_token_rel_diff"] = relative(outputs["original"], reference_tokens.float())
                    record["verify_global_rel_diff"] = relative(vectors["original__global"],
                                                                reference_global.float())
                    if max(record["verify_token_rel_diff"], record["verify_global_rel_diff"]) > 1e-3:
                        raise ValueError(f"{pid}: unmasked forward differs from official {record}")
                    verified += 1
                original = outputs["original"]
                for condition in hidden:
                    if condition == "original":
                        continue
                    # How much the hiding changed the retained tokens, per token.
                    retained = keep_both if condition == "control_masked" else keep_gtv
                    change = ((outputs[condition][retained] - original[retained]).norm(dim=1)
                              / original[retained].norm(dim=1).clamp_min(1e-12))
                    record[f"{condition}_retained_token_median_rel_change"] = float(change.median())
                    record[f"{condition}_global_rel_change"] = relative(
                        vectors[f"{condition}__global"], vectors["original__global"])
            arrays = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in vectors.items()}
            for key, value in arrays.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"{pid}: nonfinite {key}")
            if arrays["original__global"].shape != (DIMENSIONS[args.model],):
                raise ValueError(f"{pid}: global dimension {arrays['original__global'].shape}")
            target = output / "vectors" / f"{pid}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, **arrays)
            target = output / "patients" / f"{pid}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2) + "\n")
            if i % 5 == 0 or i == len(missing) or verified <= args.verify and "verify_token_rel_diff" in record:
                print(f"{args.model}: {len(ids) - len(missing) + i}/{len(ids)} {pid} "
                      + json.dumps({k: v for k, v in record.items() if k != "patient_id"}), flush=True)
            del batch, outputs, vectors
    if len(ids) == len(data) and all((output / "patients" / f"{pid}.json").exists() for pid in ids):
        pd.DataFrame([json.loads((output / "patients" / f"{pid}.json").read_text())
                      for pid in ids]).to_csv(output / "patient_summary.csv", index=False)
        (output / "COMPLETE").write_text(f"{len(ids)} patients\n")


if __name__ == "__main__":
    main()
