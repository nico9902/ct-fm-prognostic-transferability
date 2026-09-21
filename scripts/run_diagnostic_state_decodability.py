#!/usr/bin/env python3
"""Frozen Rad-JEPA scan-level diagnostic-state decodability experiments."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from extract_radjepa_embeddings import build_encoder

TASKS = ["isolation_suspicious", "nodule_ge_4mm", "emphysema", "fibrosis_or_scarring"]
REPS = ["resize32", "sliding_mean", "sliding_max", "sliding_meanmax", "sliding_gatedattention"]
SEEDS = [0, 1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--npy-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--radjepa-code-dir", type=Path, default=Path("external/RadJepa"))
    p.add_argument("--hf-model-dir", type=Path, default=Path("external/Rad-Jepa-3D-hf"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--skip-encoding", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="nlst-radjepa-diagnostic-decoding")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="pilot-1000-diagnostic-state")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def window_starts(depth):
    if depth <= 32:
        return [0]
    starts = list(range(0, depth - 32 + 1, 16))
    final = depth - 32
    if starts[-1] != final:
        starts.append(final)
    return starts


def load_volume(path):
    x = np.load(path, mmap_mode="r")
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[2:] != (256, 256):
        raise ValueError(f"bad volume shape {x.shape} at {path}")
    return np.asarray(x[0], dtype=np.float32)


def tensor_batch(arrays, device):
    return torch.from_numpy(np.stack(arrays, axis=0)[:, None]).to(device, non_blocking=True)


def encode_cache(args, manifest, out):
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    resize_path = cache / "resize32_embeddings.npy"
    window_path = cache / "sliding_window_embeddings.npy"
    window_meta_path = cache / "sliding_window_metadata.csv"
    if not args.overwrite_cache and resize_path.exists() and window_path.exists() and window_meta_path.exists():
        return
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("GPU_DEVICE 0 requested but CUDA is unavailable")
    device = torch.device(args.device)
    encoder, cfg = build_encoder(args.radjepa_code_dir, args.hf_model_dir / "config.json",
                                 args.checkpoint or args.hf_model_dir / "encoder" / "encoder.pt", device)
    resize_emb = np.zeros((len(manifest), 384), dtype=np.float32)
    all_window_emb, metas = [], []
    with torch.inference_mode():
        for base in range(0, len(manifest), args.encode_batch_size):
            chunk = manifest.iloc[base:base + args.encode_batch_size]
            arrays = [load_volume(p) for p in chunk.npy_path]
            xb = tensor_batch([np.asarray(torch.nn.functional.interpolate(
                torch.from_numpy(a)[None, None], size=(32, 256, 256), mode="trilinear", align_corners=False
            )[0, 0]) for a in arrays], device)
            resize_emb[base:base + len(chunk)] = encoder(xb).mean(1).float().cpu().numpy()
            windows = []
            local_meta = []
            for local, (row, a) in enumerate(zip(chunk.itertuples(index=False), arrays)):
                starts = window_starts(a.shape[0])
                for st in starts:
                    w = a[st:st + 32]
                    if w.shape[0] < 32:
                        w = np.pad(w, ((0, 32 - w.shape[0]), (0, 0), (0, 0)), mode="edge")
                    windows.append(w); local_meta.append((base + local, row.patient_id, row.timepoint, st, min(st + 32, a.shape[0])))
            for wb in range(0, len(windows), args.encode_batch_size):
                e = encoder(tensor_batch(windows[wb:wb + args.encode_batch_size], device)).mean(1).float().cpu().numpy()
                all_window_emb.append(e)
            metas.extend(local_meta)
            print(f"encoded {min(base + len(chunk), len(manifest))}/{len(manifest)}", flush=True)
    np.save(resize_path, resize_emb)
    np.save(window_path, np.concatenate(all_window_emb, axis=0).astype(np.float32))
    pd.DataFrame(metas, columns=["scan_index", "patient_id", "timepoint", "window_start", "window_end"]).to_csv(window_meta_path, index=False)
    (cache / "cache_metadata.json").write_text(json.dumps({"dim": int(cfg.dim), "window_size": 32, "stride": 16,
        "normalization": "prepared NPY manifest normalization", "encoder_frozen": True}, indent=2))


class GatedAttention(nn.Module):
    def __init__(self, dim=384, hidden=128):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1)
    def forward(self, x, lengths=None):
        a = self.w(self.v(x) * self.u(x)).squeeze(-1)
        if lengths is not None:
            positions = torch.arange(x.shape[1], device=x.device)[None, :]
            a = a.masked_fill(positions >= lengths[:, None], torch.finfo(a.dtype).min)
        weights = torch.softmax(a, dim=1)
        return (weights.unsqueeze(-1) * x).sum(1), weights


class GateModel(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = GatedAttention(); self.head = nn.Linear(384, 1)
    def forward(self, x, lengths):
        z, w = self.attn(x, lengths); return self.head(z).squeeze(-1), w


def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": roc_auc_score(y, prob), "auprc": average_precision_score(y, prob),
            "mcc": matthews_corrcoef(y, pred), "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def split_patients(labels, out):
    path = out / "patient_splits.csv"
    if path.exists(): return pd.read_csv(path, dtype=str)
    p = labels[["patient_id", "cancer_label"]].drop_duplicates("patient_id")
    tr, rest = train_test_split(p, test_size=.4, random_state=42, stratify=p.cancer_label)
    va, te = train_test_split(rest, test_size=.5, random_state=42, stratify=rest.cancer_label)
    s = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")], ignore_index=True)
    s.to_csv(path, index=False); return s


class DiagnosticLightningModule(pl.LightningModule):
    def __init__(self, input_dim, pos_weight, attention=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = GateModel() if attention else nn.Linear(input_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, x, lengths=None):
        return self.model(x, lengths) if self.hparams.attention else self.model(x).squeeze(-1)

    def _step(self, batch, stage):
        if self.hparams.attention:
            x, lengths, y = batch; z = self(x, lengths)[0]
        else:
            x, y = batch; z = self(x)
        loss = self.loss_fn(z, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(y))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)


def train_one(xtr, ytr, xva, yva, xte, yte, rep, seed, batch, epochs, patience,
              attention=False, wandb_config=None):
    seed_all(seed)
    if attention:
        train_ds = TensorDataset(torch.from_numpy(xtr[0]).float(), torch.from_numpy(xtr[1]).long(), torch.from_numpy(ytr).float())
        val_ds = TensorDataset(torch.from_numpy(xva[0]).float(), torch.from_numpy(xva[1]).long(), torch.from_numpy(yva).float())
        test_ds = TensorDataset(torch.from_numpy(xte[0]).float(), torch.from_numpy(xte[1]).long(), torch.from_numpy(yte).float())
        input_dim = 384
    else:
        train_ds = TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float())
        val_ds = TensorDataset(torch.from_numpy(xva).float(), torch.from_numpy(yva).float())
        test_ds = TensorDataset(torch.from_numpy(xte).float(), torch.from_numpy(yte).float())
        input_dim = xtr.shape[1]
    pos = max(float(ytr.sum()), 1.0); neg = max(float(len(ytr) - ytr.sum()), 1.0)
    module = DiagnosticLightningModule(input_dim, neg / pos, attention=attention)
    logger = None
    if wandb_config is not None:
        logger = WandbLogger(
            project=wandb_config["project"], entity=wandb_config.get("entity"),
            group=wandb_config["group"],
            name=f"{wandb_config['task']}__{rep}__seed{seed}",
            config={"task": wandb_config["task"], "representation": rep, "seed": seed,
                    "attention": attention, "pos_weight": neg / pos,
                    "epochs": epochs, "patience": patience},
        )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        max_epochs=epochs, logger=logger, enable_checkpointing=False,
        enable_progress_bar=False, deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience, min_delta=1e-5)],
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch)
    test_loader = DataLoader(test_ds, batch_size=batch)
    trainer.fit(module, train_loader, val_loader)
    predictions, weights = [], []
    module.eval()
    with torch.inference_mode():
        for batch_data in test_loader:
            if attention:
                x, lengths, _ = batch_data; z, w = module(x, lengths); weights.append(w.cpu().numpy())
            else:
                x, _ = batch_data; z = module(x)
            predictions.append(torch.sigmoid(z).cpu().numpy())
    result = metrics(yte, np.concatenate(predictions))
    if logger is not None:
        logger.log_metrics({f"test_{k}": v for k, v in result.items()})
        logger.experiment.summary.update({f"test_{k}": v for k, v in result.items()})
        logger.experiment.finish()
    return result, np.concatenate(weights) if weights else None


def main():
    a = parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(a.labels) if a.labels.suffix == ".parquet" else pd.read_csv(a.labels, dtype=str)
    nm = pd.read_csv(a.npy_manifest, dtype=str); nm = nm[nm.status.eq("prepared")].copy()
    key = ["patient_id", "timepoint", "series_uid"]
    d = labels.merge(nm[key + ["npy_path", "npy_shape", "normalization"]], on=key, how="inner", validate="one_to_one")
    d = d[d.linkage_qc.eq("ok")].copy()
    for c in TASKS: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=TASKS).reset_index(drop=True)
    splits = split_patients(d, out); d = d.merge(splits[["patient_id", "split"]], on="patient_id", validate="many_to_one")
    d.to_csv(out / "experiment_scan_manifest.csv", index=False)
    qc = {"n_patients": int(d.patient_id.nunique()), "n_scans": len(d),
          "depth": {k: float(v) for k, v in pd.to_numeric(d.npy_shape.str.extract(r"1x(\d+)x")[0]).describe()[["min", "mean", "50%", "max"]].items()},
          "n_depth_lt_32": int((pd.to_numeric(d.npy_shape.str.extract(r"1x(\d+)x")[0]) < 32).sum()),
          "windows": {"min": int(min(len(window_starts(int(x))) for x in d.npy_shape.str.extract(r"1x(\d+)x")[0])),
                      "mean": float(np.mean([len(window_starts(int(x))) for x in d.npy_shape.str.extract(r"1x(\d+)x")[0]])),
                      "max": int(max(len(window_starts(int(x))) for x in d.npy_shape.str.extract(r"1x(\d+)x")[0]))},
          "split_counts": d.groupby("split").size().to_dict(),
          "split_patients": d.groupby("split").patient_id.nunique().to_dict(),
          "label_prevalence": {c: float(d[c].mean()) for c in TASKS}}
    (out / "qc.json").write_text(json.dumps(qc, indent=2))
    if not a.skip_encoding: encode_cache(a, d, out)
    resize = np.load(out / "cache/resize32_embeddings.npy", mmap_mode="r")
    wins = np.load(out / "cache/sliding_window_embeddings.npy", mmap_mode="r")
    wm = pd.read_csv(out / "cache/sliding_window_metadata.csv")
    by_scan = {int(k): v.to_numpy() for k, v in wm.groupby("scan_index").groups.items()}
    reps = {"resize32": np.asarray(resize), "sliding_mean": np.vstack([wins[by_scan[i]].mean(0) for i in range(len(d))]),
            "sliding_max": np.vstack([wins[by_scan[i]].max(0) for i in range(len(d))])}
    reps["sliding_meanmax"] = np.hstack([reps["sliding_mean"], reps["sliding_max"]])
    gated_lengths = np.asarray([len(by_scan[i]) for i in range(len(d))], dtype=np.int64)
    max_k = int(gated_lengths.max())
    gated = np.zeros((len(d), max_k, 384), dtype=np.float32)
    for i in range(len(d)):
        z = np.asarray(wins[by_scan[i]])
        gated[i, :len(z)] = z
    reps["sliding_gatedattention"] = (gated, gated_lengths)
    rows, attn_rows = [], []
    for task in TASKS:
        for rep in REPS:
            valid = d[task].notna().to_numpy(); idx = np.flatnonzero(valid)
            tr = d.split.eq("train").to_numpy() & valid; va = d.split.eq("val").to_numpy() & valid; te = d.split.eq("test").to_numpy() & valid
            x = reps[rep]; y = d[task].to_numpy(dtype=np.float32)
            if rep != "sliding_gatedattention":
                scaler = StandardScaler().fit(x[tr]); x = scaler.transform(x).astype(np.float32)
            else:
                gated_x, lengths = x
                train_windows = np.concatenate([gated_x[i, :lengths[i]] for i in np.flatnonzero(tr)], axis=0)
                scaler = StandardScaler().fit(train_windows)
                gated_x = scaler.transform(gated_x.reshape(-1, 384)).reshape(gated_x.shape).astype(np.float32)
                x = (gated_x, lengths)
            for seed in SEEDS:
                if rep == "sliding_gatedattention":
                    train_x = (x[0][tr], x[1][tr]); val_x = (x[0][va], x[1][va]); test_x = (x[0][te], x[1][te])
                else:
                    train_x, val_x, test_x = x[tr], x[va], x[te]
                wb = {"project": a.wandb_project, "entity": a.wandb_entity,
                      "group": a.wandb_group, "task": task} if a.wandb else None
                result, weights = train_one(train_x, y[tr], val_x, y[va], test_x, y[te], rep, seed, a.train_batch_size, a.epochs, a.patience, rep == "sliding_gatedattention", wb)
                rows.append({"task": task, "representation": rep, "seed": seed, **result})
                if weights is not None:
                    test_ids = np.flatnonzero(te)
                    for j, scan_i in enumerate(test_ids):
                        for k, w in enumerate(weights[j]): attn_rows.append({"task": task, "representation": rep, "seed": seed, "scan_index": int(scan_i), "patient_id": d.iloc[scan_i].patient_id, "timepoint": d.iloc[scan_i].timepoint, "window_index": k, "attention_weight": float(w)})
    result_df = pd.DataFrame(rows); result_df.to_csv(out / "per_seed_results.csv", index=False)
    metric_cols = ["auroc", "auprc", "mcc", "accuracy", "f1", "precision", "recall", "specificity"]
    agg = result_df.groupby(["task", "representation"])[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in agg.columns]
    agg.to_csv(out / "summary_results.csv", index=False)
    pd.DataFrame(attn_rows).to_csv(out / "gated_attention_weights.csv", index=False)
    (out / "report.md").write_text("# Diagnostic state decodability\n\nFrozen Rad-JEPA-3D, patient-level splits, five-seed downstream experiments. See `qc.json`, `per_seed_results.csv`, and `summary_results.csv`.\n\nRad-JEPA was frozen; only linear heads were trained, except gated attention which learned the specified MIL aggregator plus linear head.\n")
    print(json.dumps(qc, indent=2)); print(f"results={out / 'summary_results.csv'}")


if __name__ == "__main__": main()
