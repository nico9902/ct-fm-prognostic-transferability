#!/usr/bin/env python3
"""Compare frozen Rad-JEPA linear and small nonlinear scan probes."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TASKS = ["isolation_suspicious", "nodule_ge_4mm", "emphysema", "fibrosis_or_scarring"]
REPS = ["resize32", "sliding_mean", "sliding_max", "sliding_meanmax"]
SEEDS = [0, 1, 2, 3, 4]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def metrics(y, prob):
    pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": roc_auc_score(y, prob), "auprc": average_precision_score(y, prob),
            "mcc": matthews_corrcoef(y, pred), "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


class Probe(nn.Module):
    def __init__(self, dim, nonlinear):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))
                    if nonlinear else nn.Linear(dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_probe(xtr, ytr, xva, yva, xte, yte, seed, nonlinear, epochs, patience, batch):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Probe(xtr.shape[1], nonlinear).to(device)
    pos = max(float(ytr.sum()), 1.0); neg = max(float(len(ytr) - ytr.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float())
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    xtv, ytv = torch.from_numpy(xva).float().to(device), torch.from_numpy(yva).float().to(device)
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            z = model(xb.to(device)); loss = loss_fn(z, yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            vloss = float(loss_fn(model(xtv), ytv).cpu())
        if vloss < best_loss - 1e-5:
            best_loss, stale, best_state = vloss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prob = torch.sigmoid(model(torch.from_numpy(xte).float().to(device))).cpu().numpy()
    return metrics(yte, prob), int(_ + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="diagnostic_decodability artifact directory")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.root / "experiment_scan_manifest.csv", dtype=str)
    for c in TASKS: d[c] = pd.to_numeric(d[c], errors="coerce")
    splits = pd.read_csv(a.root / "patient_splits.csv", dtype=str)
    d = d.merge(splits[["patient_id", "split"]], on="patient_id", how="left", validate="many_to_one", suffixes=("", "_split"))
    wins = np.load(a.root / "cache/sliding_window_embeddings.npy", mmap_mode="r")
    wm = pd.read_csv(a.root / "cache/sliding_window_metadata.csv")
    by_scan = {int(k): v.to_numpy() for k, v in wm.groupby("scan_index").groups.items()}
    resize = np.asarray(np.load(a.root / "cache/resize32_embeddings.npy", mmap_mode="r"))
    reps = {"resize32": resize,
            "sliding_mean": np.vstack([wins[by_scan[i]].mean(0) for i in range(len(d))]),
            "sliding_max": np.vstack([wins[by_scan[i]].max(0) for i in range(len(d))])}
    reps["sliding_meanmax"] = np.hstack([reps["sliding_mean"], reps["sliding_max"]])
    rows = []
    for task in TASKS:
        valid = d[task].notna().to_numpy(); y = d[task].to_numpy(dtype=np.float32)
        tr = d.split.eq("train").to_numpy() & valid; va = d.split.eq("val").to_numpy() & valid; te = d.split.eq("test").to_numpy() & valid
        for rep in REPS:
            x = reps[rep]; scaler = StandardScaler().fit(x[tr]); x = scaler.transform(x).astype(np.float32)
            for nonlinear in [False, True]:
                kind = "mlp" if nonlinear else "linear"
                for seed in SEEDS:
                    result, epochs = fit_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], seed, nonlinear, a.epochs, a.patience, a.batch)
                    rows.append({"task": task, "representation": rep, "model": kind, "seed": seed, "epochs_run": epochs, **result})
                    print(task, rep, kind, seed, f"AUROC={result['auroc']:.4f}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(a.output_dir / "per_seed_results.csv", index=False)
    cols = ["auroc", "auprc", "mcc", "accuracy", "f1", "precision", "recall", "specificity"]
    agg = out.groupby(["task", "representation", "model"])[cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in agg.columns]
    agg.to_csv(a.output_dir / "summary_results.csv", index=False)
    (a.output_dir / "metadata.json").write_text(json.dumps({"tasks": TASKS, "representations": REPS, "seeds": SEEDS,
        "hidden": 128, "dropout": .2, "patient_split_source": str(a.root / "patient_splits.csv"),
        "encoder_frozen": True, "cache_source": str(a.root / "cache")}, indent=2))


if __name__ == "__main__": main()
