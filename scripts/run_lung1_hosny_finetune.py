#!/usr/bin/env python3
"""Hosny et al. 2018 CNN fine-tuned for LUNG1 survival (phase 2.3, tailored deep method).

Selected-checkpoint protocol of E1/E2: repetition 0, five outer folds, inner fold 0 of
each outer development set as validation; one test pass with the epoch of maximal
validation Uno C (earliest tie).

Initialised from the HarvardRT weights. Risk = logit(dead) - logit(alive), so epoch 0
is the zero-shot model. Exact full-batch Efron partial likelihood on all training
patients every epoch (as in the LoRA/partial rungs: risks without gradient, dLoss/dRisk,
then chunked backward of risk * derivative, which gives the exact full-batch gradient
because the network is deterministic in eval mode), no dropout, frozen BatchNorm
statistics (all weights and BN affine parameters trained), AdamW. FP32 by default (runs
on a V100); --bf16 enables BF16 autocast. Patches are built as in
extract_lung1_tumor_patch_embeddings.py and cached.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from extract_lung1_tumor_patch_embeddings import centroid, hosny_patch, paths
from hosny_deep_prognosis import load as load_hosny
from run_lung1_e2_encoder_bf16_lightning import aggregate, write_json
from run_lung1_e2_mlp import EfronLoss, breslow_survival
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, GRID, ROOT, cohort, fold_metrics, inner_assignments, locked_folds, sha256, uno_c

OUTPUT = ROOT / "outputs/lung1_bf16_selected/finetune/hosny"


PATCHES = ROOT / "outputs/lung1_tumor_patches/hosny_patches.npy"
CHUNK = 16
BF16 = False


def forward(net, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        logits = net(x).float()
    return logits[:, 0] - logits[:, 1]


def risks(net, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([forward(net, x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)]).cpu().numpy()


def step(net, optimizer, objective, x: torch.Tensor) -> float:
    detached = torch.tensor(risks(net, x), device=x.device, requires_grad=True)
    loss = objective(detached)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training Cox loss")
    derivative = torch.autograd.grad(loss, detached)[0].detach()
    optimizer.zero_grad(set_to_none=True)
    for i in range(0, len(x), CHUNK):
        (forward(net, x[i:i + CHUNK]) * derivative[i:i + CHUNK]).sum().backward()
    optimizer.step()
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    global BF16
    BF16 = args.bf16
    device = torch.device("cuda")
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    inner = inner_assignments(data, folds)
    if PATCHES.exists():
        patches = np.load(PATCHES)
    else:
        patches = np.stack([hosny_patch(sitk.ReadImage(str(paths(pid)[0])), centroid(paths(pid)[1]))
                            for pid in data.patient_id]).astype(np.float32)
        PATCHES.parent.mkdir(parents=True, exist_ok=True)
        np.save(PATCHES, patches)
        (PATCHES.parent / "hosny_patches_order.txt").write_text("\n".join(data.patient_id) + "\n")
    if len(patches) != len(data):
        raise ValueError("Cached patches do not match the cohort")
    x_all = torch.from_numpy(patches).to(device)
    t = data.os_time_days.to_numpy(float)
    e = data.os_event.to_numpy(int)
    write_json(args.output / "protocol.json", {
        "model": "hosny", "method": "finetune", "init": "HarvardRT weights (modelhub deep-prognosis)",
        "risk": "logit_dead - logit_alive", "loss": "full-batch Efron (chunked exact gradient), FP32",
        "precision": "bf16 autocast" if BF16 else "fp32", "device": torch.cuda.get_device_name(0),
        "train_mode": "eval (no dropout, frozen BN statistics)", "optimizer": "AdamW",
        "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
        "selection": "maximum_validation_uno_c_36m_earliest_tie (epoch 0 = zero-shot)",
        "outer_repetition": 0, "validation_inner_fold": 0,
        "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS)})
    for fold in range(5):
        if (args.output / "folds" / f"fold_{fold}.json").exists():
            continue
        started = time.monotonic()
        torch.manual_seed(20261005 + fold)
        test = data.patient_id.isin(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"]).to_numpy()
        val = data.patient_id.isin(inner.loc[(inner.repetition == 0) & (inner.fold == fold)
                                             & (inner.inner_fold == 0), "patient_id"]).to_numpy()
        train = ~(test | val)
        net = load_hosny().to(device).eval()
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        objective = EfronLoss(t[train], e[train])
        x_train, x_val, x_test = x_all[train], x_all[val], x_all[test]
        best = (-np.inf, 0, {k: v.detach().clone() for k, v in net.state_dict().items()})
        history = []
        for epoch in range(args.epochs + 1):
            if epoch > 0:
                step(net, optimizer, objective, x_train)
            train_risk = risks(net, x_train)
            val_risk = risks(net, x_val)
            score = uno_c(t[train], e[train], t[val], e[val], val_risk)
            history.append({"epoch": epoch, "val_uno_c_36m": float(score),
                            "train_uno_c_36m": float(uno_c(t[train], e[train], t[train], e[train], train_risk))})
            if np.isfinite(score) and score > best[0]:
                best = (float(score), epoch, {k: v.detach().clone() for k, v in net.state_dict().items()})
        net.load_state_dict(best[2])
        train_risk = risks(net, x_train)
        risk = risks(net, x_test)
        survival = breslow_survival(t[train], e[train], train_risk, risk)
        metrics = fold_metrics(t[train], e[train], t[test], e[test], risk, survival)
        row = {"model": "hosny", "method": "finetune", "repetition": 0, "fold": fold,
               "n_train": int(train.sum()), "n_validation": int(val.sum()), "n_test": int(test.sum()),
               "elapsed_seconds": time.monotonic() - started, "selected_epoch": best[1],
               "selected_validation_uno_c_36m": best[0],
               "trainable_parameters": sum(p.numel() for p in net.parameters()), **metrics}
        predictions = []
        for i, index in enumerate(np.flatnonzero(test)):
            prediction = {"model": "hosny", "method": "finetune", "repetition": 0, "fold": fold,
                          "patient_id": data.patient_id.iloc[index], "os_time_days": float(t[index]),
                          "os_event": int(e[index]), "risk": float(risk[i]),
                          "risk_36m": float(-np.log(np.clip(survival[i, -1], 1e-12, 1.0)))}
            prediction.update({f"survival_{j:02d}": float(v) for j, v in enumerate(survival[i])})
            predictions.append(prediction)
        write_json(args.output / "folds" / f"fold_{fold}.json", {"metrics": row, "predictions": predictions,
                                                                "history": history})
        aggregate(args.output, data)
        print(f"hosny finetune fold={fold}: best epoch {best[1]} val={best[0]:.4f} "
              f"test Uno36={metrics['uno_c_36m']:.4f}", flush=True)


if __name__ == "__main__":
    main()
