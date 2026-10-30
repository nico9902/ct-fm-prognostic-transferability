#!/usr/bin/env python3
"""Image-level E2 LoRA or last-block partial fine-tuning with exact Cox risk sets.

Uses fixed, predeclared hyperparameters and the locked E1 5x5 outer folds.
The two-pass update recomputes each patient's score to obtain an exact full-risk-set
gradient without keeping every encoder graph resident at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_lung1_e1_embeddings import CHECKPOINTS, DIMENSIONS, load_encoder  # noqa: E402
from run_lung1_e2_mlp import EfronLoss, breslow_survival  # noqa: E402
from run_lung1_frozen_cox_e1 import (  # noqa: E402
    GRID, OUTER_FOLDS, REPETITIONS, cohort, fold_metrics, locked_folds, sha256,
)

MODELS = ("tangerine", "radjepa", "colipri", "ctclip", "dinov3")
METHODS = ("lora", "partial")
SETTINGS = {"lora": {"rank": 8, "alpha": 16, "lr": 1e-4, "weight_decay": 1e-4},
            "partial": {"last_blocks": 1, "lr": 1e-5, "weight_decay": 1e-4}}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x))

    @property
    def weight(self):
        # Some encoder kernels read Linear.weight directly rather than calling it.
        return self.base.weight + self.scale * (self.b.weight @ self.a.weight)

    @property
    def bias(self):
        return self.base.bias


def last_blocks(name: str, model: nn.Module) -> list[nn.Module]:
    if name == "tangerine":
        return [model.blocks[-1]]
    if name == "radjepa":
        return [model.blocks[-1]]
    if name == "colipri":
        return [model.image_encoder.backbone.eva.blocks[-1]]
    if name == "dinov3":
        return [model.layer[-1]]
    # CT-CLIP's quantizer separates its spatial/temporal blocks from the
    # exported visual embedding. Adapt its final visual projection instead.
    return []


def attach_lora(module: nn.Module, rank: int, alpha: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            replaced += 1
        else:
            replaced += attach_lora(child, rank, alpha)
    return replaced


class AdaptedRisk(nn.Module):
    def __init__(self, name: str, encoder: nn.Module, method: str):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.encoder.requires_grad_(False).eval()
        targets = last_blocks(name, encoder)
        if name == "ctclip":
            if method == "lora":
                self.encoder["projection"] = LoRALinear(
                    self.encoder["projection"], SETTINGS[method]["rank"], SETTINGS[method]["alpha"])
                self.adapted_linear_count = 1
            else:
                self.encoder["projection"].requires_grad_(True)
                self.adapted_linear_count = 0
        elif method == "lora":
            count = sum(attach_lora(target, SETTINGS[method]["rank"], SETTINGS[method]["alpha"])
                        for target in targets)
            if count == 0:
                raise ValueError(f"{name}: no linear modules found in selected last block(s)")
            self.adapted_linear_count = count
        else:
            for target in targets:
                target.requires_grad_(True)
            self.adapted_linear_count = 0
        self.norm = nn.LayerNorm(DIMENSIONS[name])
        self.head = nn.Linear(DIMENSIONS[name], 1)
        self.eval()  # deterministic two-pass gradient; trainable parameters still get gradients

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.name == "tangerine":
            embedding = self.encoder.fc_norm(self.encoder.forward_features(batch))
        elif self.name == "radjepa":
            embedding = self.encoder(batch, indices=None).mean(dim=1)
        elif self.name == "colipri":
            embedding = self.encoder.encode_image(batch, pool=True, project=True, normalize=False)
        elif self.name == "dinov3":
            from extract_lung1_e1_embeddings import dinov3_forward
            embedding = dinov3_forward(self.encoder, batch)[:, 0].float().mean(dim=0, keepdim=True)
        else:
            tokens = self.encoder["encoder"](batch, return_encoded_tokens=True)
            pooled = tokens.mean(dim=1).reshape(len(batch), -1)
            embedding = torch.nn.functional.normalize(self.encoder["projection"](pooled), dim=-1)
        return self.head(self.norm(embedding)).flatten()


def load_inputs(name: str, data: pd.DataFrame) -> tuple[list[Path], dict]:
    folder = ROOT / f"outputs/lung1_e2_inputs/{name}/original_ct"
    path = folder / "input_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Preprocess image inputs first: {path}")
    manifest = pd.read_csv(path, dtype={"patient_id": str, "ct_series_uid": str})
    if manifest.patient_id.duplicated().any() or set(manifest.patient_id) != set(data.patient_id):
        raise ValueError("Image inputs must cover the exact locked cohort")
    manifest = manifest.set_index("patient_id").loc[data.patient_id]
    if not np.array_equal(manifest.ct_series_uid.to_numpy(), data.ct_series_uid.to_numpy()):
        raise ValueError("Image input CT series UIDs differ from the locked cohort")
    paths = [folder / filename for filename in manifest.input_path]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Image input cache contains missing files")
    return paths, {"manifest_sha256": sha256(path),
                   "input_protocol_sha256": sha256(folder / "protocol.json")}


def single_score(net: AdaptedRisk, path: Path, device: torch.device) -> torch.Tensor:
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    image = torch.as_tensor(np.asarray(array, dtype=np.float32).copy(), device=device)[None, None]
    return net(image)[0]


def scores_no_grad(net: AdaptedRisk, paths: list[Path], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return np.array([float(single_score(net, path, device)) for path in paths])


def train_epoch(net: AdaptedRisk, optimizer: torch.optim.Optimizer,
                paths: list[Path], time_: np.ndarray, event: np.ndarray,
                device: torch.device) -> float:
    # First pass builds the exact full risk set while storing no encoder graphs.
    risk = torch.tensor(scores_no_grad(net, paths, device), device=device,
                        dtype=torch.float32, requires_grad=True)
    loss = EfronLoss(time_, event)(risk)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite full-risk-set Efron loss")
    derivative = torch.autograd.grad(loss, risk)[0].detach()
    optimizer.zero_grad(set_to_none=True)
    # The model remains in eval mode, making the recomputed scores deterministic.
    for index, path in enumerate(paths):
        (single_score(net, path, device) * derivative[index]).backward()
    optimizer.step()
    return float(loss.detach())


def smoke(name: str, method: str, device: torch.device):
    encoder, _, _ = load_encoder(name, device, CHECKPOINTS[name])
    net = AdaptedRisk(name, encoder, method).to(device)
    from extract_lung1_e1_embeddings import (read_image, tangerine_input, radjepa_input,
                                             ctclip_input, dinov3_input)
    data = cohort(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv")
    image = read_image(data.iloc[0], "original_ct", ROOT / "data/processed/lung1/thorax_hu")
    if name == "tangerine":
        batch = tangerine_input(image).to(device)
    elif name == "radjepa":
        batch = radjepa_input(image).to(device)
    elif name == "ctclip":
        batch = ctclip_input(image, device)
    elif name == "dinov3":
        batch = dinov3_input(image).to(device)
    else:
        import torchio as tio
        from colipri import get_processor
        processor = get_processor(image_only=True)
        batch = processor.to_images_batch(processor.process_images(tio.ScalarImage.from_sitk(image))).to(device)
    risk = net(batch)
    if not torch.isfinite(risk).all():
        raise ValueError("Nonfinite adapted risk")
    risk.sum().backward()
    trainable = [p for p in net.parameters() if p.requires_grad]
    if not trainable or not any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
        raise ValueError("No finite adaptation gradients")
    adapted = [p for p in net.encoder.parameters() if p.requires_grad]
    if not adapted or not any(p.grad is not None and torch.isfinite(p.grad).all()
                              and float(p.grad.abs().sum()) > 0 for p in adapted):
        raise ValueError("No nonzero gradient reached the selected encoder block")
    print(json.dumps({"model": name, "method": method, "risk": float(risk[0]),
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "adapted_linear_count": net.adapted_linear_count,
                      "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/lung1_e2_encoder_adaptation")
    parser.add_argument("--wandb-project", default="lung1-prognostic-transferability")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_folds < 0:
        parser.error("epochs must be positive and max-folds nonnegative")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Image-level encoder adaptation requires CUDA")
    device = torch.device(args.device)
    if args.smoke_only:
        smoke(args.model, args.method, device)
        return
    data = cohort(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv")
    folds = locked_folds(data, ROOT / "configs/lung1_e1_folds.csv")
    paths, input_provenance = load_inputs(args.model, data)
    output = args.output_root / args.method / args.model
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"method": args.method, "model": args.model, "settings": SETTINGS[args.method],
                "epochs": args.epochs, "max_folds": args.max_folds,
                "cohort_sha256": sha256(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"),
                "folds_sha256": sha256(ROOT / "configs/lung1_e1_folds.csv"),
                "checkpoint_sha256": sha256(CHECKPOINTS[args.model]),
                "inputs": input_provenance,
                "risk_set": "exact_full_training_fold_efron_two_pass_gradient",
                "selection": "fixed_predeclared_hyperparameters_no_test_tuning"}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a new output root")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    metric_path = output / "fold_metrics.csv"
    prediction_path = output / "out_of_fold_predictions.csv"
    metrics = pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else []
    predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    if bool(metrics) != bool(predictions):
        raise ValueError("Metrics and predictions must both exist for resumption")
    completed = {(int(row["repetition"]), int(row["fold"])) for row in metrics}
    if len(completed) != len(metrics):
        raise ValueError("Duplicate completed fold")
    import wandb
    run_id = hashlib.sha1(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:8]
    run = wandb.init(project=args.wandb_project, name=f"E2-{args.method}-{args.model}",
                     group="lung1-e2-original-ct", job_type=f"{args.method}-cox",
                     mode=os.environ.get("WANDB_MODE", "offline"),
                     dir=str(output), config=protocol, id=run_id, resume="allow")
    started = time.monotonic()
    try:
        for repetition in range(REPETITIONS):
            for fold in range(OUTER_FOLDS):
                if (repetition, fold) in completed:
                    continue
                if args.max_folds and len(metrics) >= args.max_folds:
                    break
                test_ids = set(folds.loc[(folds.repetition == repetition) &
                                         (folds.fold == fold), "patient_id"])
                test_mask = data.patient_id.isin(test_ids).to_numpy()
                train_idx = np.flatnonzero(~test_mask)
                test_idx = np.flatnonzero(test_mask)
                train_time = data.os_time_days.to_numpy(dtype=float)[train_idx]
                train_event = data.os_event.to_numpy(dtype=int)[train_idx]
                test_time = data.os_time_days.to_numpy(dtype=float)[test_idx]
                test_event = data.os_event.to_numpy(dtype=int)[test_idx]
                torch.manual_seed(20261005 + repetition * 100 + fold)
                encoder, _, _ = load_encoder(args.model, device, CHECKPOINTS[args.model])
                net = AdaptedRisk(args.model, encoder, args.method).to(device)
                parameters = [p for p in net.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=SETTINGS[args.method]["lr"],
                                              weight_decay=SETTINGS[args.method]["weight_decay"])
                torch.cuda.reset_peak_memory_stats(device)
                train_paths = [paths[i] for i in train_idx]
                for epoch in range(args.epochs):
                    train_loss = train_epoch(net, optimizer, train_paths, train_time, train_event, device)
                    wandb.log({"repetition": repetition, "fold": fold, "epoch": epoch,
                               "train_loss": train_loss})
                train_risk = scores_no_grad(net, train_paths, device)
                risk = scores_no_grad(net, [paths[i] for i in test_idx], device)
                survival = breslow_survival(train_time, train_event, train_risk, risk)
                score = fold_metrics(train_time, train_event, test_time, test_event, risk, survival)
                row = {"model": args.model, "method": args.method,
                       "repetition": repetition, "fold": fold,
                       "trainable_parameters": sum(p.numel() for p in parameters),
                       "total_parameters": sum(p.numel() for p in net.parameters()),
                       "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                       "elapsed_seconds": time.monotonic() - started,
                       "train_loss": train_loss, **score}
                row["trainable_fraction"] = row["trainable_parameters"] / row["total_parameters"]
                metrics.append(row)
                for i, index in enumerate(test_idx):
                    pred = {"model": args.model, "method": args.method,
                            "repetition": repetition, "fold": fold,
                            "patient_id": data.patient_id.iloc[index],
                            "os_time_days": test_time[i], "os_event": test_event[i], "risk": risk[i]}
                    pred.update({f"survival_{j:02d}": survival[i, j] for j in range(len(GRID))})
                    predictions.append(pred)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
                pd.DataFrame(metrics).to_csv(metric_path, index=False)
                wandb.log({"repetition": repetition, "fold": fold,
                           **{key: float(value) for key, value in score.items()},
                           "trainable_parameters": row["trainable_parameters"],
                           "total_parameters": row["total_parameters"],
                           "trainable_fraction": row["trainable_fraction"],
                           "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"],
                           "elapsed_seconds": row["elapsed_seconds"]})
                print(f"{args.method}/{args.model} repeat={repetition} fold={fold} "
                      f"Uno C={score['uno_c_36m']:.3f}", flush=True)
                del net, encoder, optimizer
                torch.cuda.empty_cache()
        frame = pd.DataFrame(metrics)
        summary = {key: float(frame[key].mean()) for key in
                   ("uno_c_36m", "auc_1y", "auc_2y", "auc_3y", "ibs_3m_36m",
                    "calibration_in_large_2y", "calibration_slope")}
        summary.update({"completed_folds": len(metrics), "expected_folds": REPETITIONS * OUTER_FOLDS})
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        wandb.summary.update(summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
