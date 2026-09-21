#!/usr/bin/env python3
"""Train and evaluate a frozen-COLIPRI next-state latent world model.

The script deliberately keeps the encoder out of the loop.  It consumes the
existing 864-d mean-pooled COLIPRI cache, the existing patient split, and the
existing prognostic/diagnostic labels.  Every scaler is fitted on train
patients only; the pair split is patient-level, so both transitions from a
patient remain in one split.
"""
from __future__ import annotations

import argparse
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

try:
    import wandb
except ImportError:  # pragma: no cover - the project declares wandb
    wandb = None

HORIZONS = ["1y", "2y", "3y"]
DIAGNOSTIC_TASKS = ["isolation_suspicious", "nodule_ge_4mm", "emphysema", "fibrosis_or_scarring"]
SEEDS = [0, 1, 2, 3, 4]
DIM = 864


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualLatentMLP(nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.z_norm = nn.LayerNorm(dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.mlp = nn.Sequential(
            nn.Linear(dim + 32, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.z_norm(z_t), self.time_mlp(dt)], dim=-1)
        return z_t + self.mlp(h)


def loss_parts(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = torch.mean((pred - target) ** 2)
    cosine_loss = (1.0 - nn.functional.cosine_similarity(pred, target, dim=-1)).mean()
    return mse + cosine_loss, mse, cosine_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--embedding-cache", type=Path, default=Path("outputs/colipri_decodability/mean/cache"))
    p.add_argument("--prognostic-labels", type=Path, default=Path("data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv"))
    p.add_argument("--diagnostic-manifest", type=Path, default=Path("outputs/colipri_decodability/experiment_scan_manifest.csv"))
    p.add_argument("--patient-splits", type=Path, default=Path("outputs/colipri_decodability/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/colipri_latent_world_model"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--wandb-project", default="nlst-colipri-latent-world-model")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="colipri-next-state-baseline")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def build_pairs(a: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    z = np.load(a.embedding_cache / "embeddings.npy", mmap_mode="r")
    em = pd.read_csv(a.embedding_cache / "embedding_metadata.csv", dtype=str)
    prog = pd.read_csv(a.prognostic_labels, dtype=str).drop(columns=["screening_round"])
    diag = pd.read_csv(a.diagnostic_manifest, dtype=str)
    if "screening_round" not in diag.columns and "timepoint" in diag.columns:
        diag = diag.rename(columns={"timepoint": "screening_round"})
    splits = pd.read_csv(a.patient_splits, dtype=str)[["patient_id", "split"]]
    diag_cols = ["patient_id", "screening_round", "series_uid"] + DIAGNOSTIC_TASKS
    diag = diag[diag_cols].drop_duplicates(["patient_id", "screening_round", "series_uid"])
    m = em.merge(prog, on=["patient_id", "series_uid"], how="left", validate="one_to_one")
    m = m.merge(diag, on=["patient_id", "screening_round", "series_uid"], how="left", validate="one_to_one")
    for c in ["scan_day_from_randomization", "cancer_event", "cancer_time_days", "future_cancer_1y", "future_cancer_2y", "future_cancer_3y"] + DIAGNOSTIC_TASKS:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["embedding_index"] = m["embedding_index"].astype(int)
    fields = ["embedding_index", "scan_day_from_randomization", "cancer_event", "cancer_time_days", "future_cancer_1y", "future_cancer_2y", "future_cancer_3y", "prognostic_label_quality_flags"] + DIAGNOSTIC_TASKS
    wide = m.set_index(["patient_id", "screening_round"])[fields].unstack("screening_round")

    def make(t: str, t1: str) -> pd.DataFrame:
        avail = wide[("embedding_index", t)].notna() & wide[("embedding_index", t1)].notna()
        idx = wide.index[avail]
        out = pd.DataFrame({"patient_id": idx, "pair_type": f"{t}_{t1}",
                            "idx_t": wide[("embedding_index", t)][avail].astype(int).to_numpy(),
                            "idx_t1": wide[("embedding_index", t1)][avail].astype(int).to_numpy(),
                            "dt_days": (wide[("scan_day_from_randomization", t1)][avail] - wide[("scan_day_from_randomization", t)][avail]).to_numpy()})
        for c in ["cancer_event", "cancer_time_days", "future_cancer_1y", "future_cancer_2y", "future_cancer_3y"]:
            out[f"primary_{c}"] = wide[(c, t1)][avail].to_numpy()
            out[f"secondary_{c}"] = wide[(c, t)][avail].to_numpy()
        for c in DIAGNOSTIC_TASKS:
            out[f"diag_t_{c}"] = wide[(c, t)][avail].to_numpy()
            out[f"diag_t1_{c}"] = wide[(c, t1)][avail].to_numpy()
        return out

    pairs = pd.concat([make("T0", "T1"), make("T1", "T2")], ignore_index=True)
    pairs = pairs.merge(splits, on="patient_id", how="left", validate="many_to_one")
    if pairs.split.isna().any():
        raise ValueError("Some embedded patients are missing from the existing patient split")
    if pairs.dt_days.isna().any() or (pairs.dt_days <= 0).any():
        raise ValueError("Invalid longitudinal time deltas")
    return pairs, np.asarray(z, dtype=np.float32)


def latent_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    return {"mse": float(np.mean(diff ** 2)), "cosine_similarity": float(np.mean(np.sum(pred * target, 1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1) + 1e-12))), "l2_distance": float(np.mean(np.linalg.norm(diff, axis=1)))}


def smoke(model: nn.Module, device: torch.device, dim: int = DIM) -> None:
    zt, dt, zt1 = torch.randn(2, dim, device=device), torch.randn(2, 1, device=device), torch.randn(2, dim, device=device)
    pred = model(zt, dt)
    total, mse, cos = loss_parts(pred, zt1)
    print(json.dumps({"z_t_shape": list(zt.shape), "dt_shape": list(dt.shape), "z_hat_t1_shape": list(pred.shape), "z_t1_shape": list(zt1.shape), "loss": float(total), "mse_loss": float(mse), "cosine_loss": float(cos)}, indent=2), flush=True)


def train_seed(a, pairs, z, seed, device):
    seed_all(seed)
    out = a.output_dir
    tr = pairs.split.eq("train").to_numpy(); va = pairs.split.eq("val").to_numpy(); te = pairs.split.eq("test").to_numpy()
    dt_scaler = StandardScaler().fit(pairs.loc[tr, ["dt_days"]])
    dt = dt_scaler.transform(pairs[["dt_days"]]).astype(np.float32)
    zt = z[pairs.idx_t.to_numpy()]; zt1 = z[pairs.idx_t1.to_numpy()]
    train_ds = TensorDataset(torch.from_numpy(zt[tr]), torch.from_numpy(dt[tr]), torch.from_numpy(zt1[tr]))
    val_ds = TensorDataset(torch.from_numpy(zt[va]), torch.from_numpy(dt[va]), torch.from_numpy(zt1[va]))
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    vloader = DataLoader(val_ds, batch_size=a.batch_size)
    model = ResidualLatentMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    best = (float("inf"), None, 0); stale = 0
    history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); sums = np.zeros(3); n = 0
        for xb, db, yb in loader:
            xb, db, yb = xb.to(device), db.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); total, mse, cos = loss_parts(model(xb, db), yb); total.backward(); opt.step()
            sums += np.array([float(total), float(mse), float(cos)]) * len(xb); n += len(xb)
        model.eval(); vs = np.zeros(3); vn = 0
        with torch.inference_mode():
            for xb, db, yb in vloader:
                total, mse, cos = loss_parts(model(xb.to(device), db.to(device)), yb.to(device)); vs += np.array([float(total), float(mse), float(cos)]) * len(xb); vn += len(xb)
        row = {"seed": seed, "epoch": epoch, "train_loss": sums[0]/n, "train_mse": sums[1]/n, "train_cosine_loss": sums[2]/n, "val_loss": vs[0]/vn, "val_mse": vs[1]/vn, "val_cosine_loss": vs[2]/vn}; history.append(row)
        if wandb.run is not None: wandb.log(row)
        if row["val_loss"] < best[0] - 1e-6:
            best = (row["val_loss"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch); stale = 0
        else: stale += 1
        if stale >= a.patience: break
    model.load_state_dict(best[1]); torch.save({"model_state_dict": model.state_dict(), "seed": seed, "best_epoch": best[2], "dt_scaler_mean": dt_scaler.mean_, "dt_scaler_scale": dt_scaler.scale_}, out / "checkpoints" / f"residual_mlp_seed{seed}.pt")
    model.eval()
    with torch.inference_mode():
        pred = model(torch.from_numpy(zt).to(device), torch.from_numpy(dt).to(device)).cpu().numpy()
    pd.DataFrame(history).to_csv(out / "training_history" / f"seed{seed}.csv", index=False)
    np.save(out / "predicted_embeddings" / f"seed{seed}.npy", pred)
    return pred, {"seed": seed, "best_epoch": best[2], "best_val_loss": best[0], "n_epochs": len(history)}


def probe_metrics(y, prob):
    pred = (prob >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": roc_auc_score(y, prob), "auprc": average_precision_score(y, prob), "mcc": matthews_corrcoef(y, pred), "accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "specificity": tn/(tn+fp) if tn+fp else np.nan}


def run_probe(a, pairs, reps, prefix, label_cols, out_name, device):
    rows = []
    for scope, mask in [("pooled", np.ones(len(pairs), bool)), ("T0_T1", pairs.pair_type.eq("T0_T1").to_numpy()), ("T1_T2", pairs.pair_type.eq("T1_T2").to_numpy())]:
        for label in label_cols:
            valid = mask & pairs[label].notna().to_numpy(); sub = pairs.loc[valid].reset_index(drop=True); y = sub[label].to_numpy(float)
            split = sub.split.to_numpy(); train, val, test = split == "train", split == "val", split == "test"
            if y[train].sum() == 0 or y[test].sum() == 0 or len(np.unique(y[train])) < 2: continue
            for rep_name, X in reps.items():
                scaler = StandardScaler().fit(X[valid][train]); Xs = scaler.transform(X[valid]).astype(np.float32)
                for seed in a.seeds:
                    seed_all(seed); clf = nn.Linear(Xs.shape[1], 1).to(device); pos = max(y[train].sum(), 1); neg = max(train.sum()-y[train].sum(), 1); loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], device=device)); opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
                    trds = TensorDataset(torch.from_numpy(Xs[train]), torch.from_numpy(y[train].astype(np.float32))); vds = TensorDataset(torch.from_numpy(Xs[val]), torch.from_numpy(y[val].astype(np.float32)))
                    best, state, stale = float("inf"), None, 0
                    for _ in range(100):
                        clf.train()
                        for xb, yb in DataLoader(trds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed)):
                            opt.zero_grad(); l = loss_fn(clf(xb.to(device)).squeeze(1), yb.to(device)); l.backward(); opt.step()
                        clf.eval()
                        with torch.inference_mode(): vl = float(loss_fn(clf(torch.from_numpy(Xs[val]).to(device)).squeeze(1), torch.from_numpy(y[val].astype(np.float32)).to(device)))
                        if vl < best - 1e-5: best, state, stale = vl, {k: v.detach().cpu().clone() for k,v in clf.state_dict().items()}, 0
                        else: stale += 1
                        if stale >= 15: break
                    clf.load_state_dict(state); clf.eval()
                    with torch.inference_mode(): prob = torch.sigmoid(clf(torch.from_numpy(Xs[test]).to(device)).squeeze(1)).cpu().numpy()
                    rows.append({"scope": scope, "label": label, "representation": rep_name, "seed": seed, "n_test": int(test.sum()), "n_test_positive": int(y[test].sum()), **probe_metrics(y[test], prob)})
    pd.DataFrame(rows).to_csv(a.output_dir / out_name, index=False)
    if rows:
        pd.DataFrame(rows).groupby(["scope", "label", "representation"])[["auroc", "auprc", "mcc", "f1"]].agg(["mean", "std"]).to_csv(a.output_dir / out_name.replace("per_seed", "summary").replace(".csv", "_summary.csv"))


def main() -> None:
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    for d in ["checkpoints", "training_history", "predicted_embeddings"]: (a.output_dir / d).mkdir(exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    pairs, z = build_pairs(a); pairs.to_csv(a.output_dir / "pairs_manifest.csv", index=False)
    pairs["dt_days"].describe().to_json(a.output_dir / "time_delta_summary.json")
    smoke(ResidualLatentMLP().to(device), device)
    if a.smoke_only: return
    with torch.inference_mode(): identity = z[pairs.idx_t.to_numpy()]
    np.save(a.output_dir / "predicted_embeddings" / "identity.npy", identity)
    all_preds = {"identity": identity}; train_rows = []
    for seed in a.seeds:
        run = None
        if wandb is not None and not a.no_wandb:
            run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, group=a.wandb_group, name=f"residual_mlp_seed{seed}", config={"seed": seed, "architecture": "864->1024->1024->864 residual", "loss": "mse+cosine_equal", "dropout": .1, "pairs": len(pairs), "device": str(device)}, reinit="finish_previous", settings=wandb.Settings(root_dir=str(a.output_dir / "wandb"), x_disable_stats=True))
        pred, meta = train_seed(a, pairs, z, seed, device); all_preds[f"learned_seed{seed}"] = pred; train_rows.append(meta)
        if run is not None: run.summary.update(meta); run.finish()
    pd.DataFrame(train_rows).to_csv(a.output_dir / "training_per_seed_metrics.csv", index=False)
    learned = np.mean(np.stack([all_preds[f"learned_seed{s}"] for s in a.seeds]), axis=0); np.save(a.output_dir / "predicted_embeddings" / "learned_mean.npy", learned); all_preds["learned"] = learned
    latent_rows = []
    for scope, mask in [("pooled", np.ones(len(pairs), bool)), ("T0_T1", pairs.pair_type.eq("T0_T1").to_numpy()), ("T1_T2", pairs.pair_type.eq("T1_T2").to_numpy())]:
        for rep in ["identity"] + [f"learned_seed{s}" for s in a.seeds] + ["learned"]:
            m = mask & pairs.split.eq("test").to_numpy(); row = {"scope": scope, "representation": rep, **latent_metrics(all_preds[rep][m], z[pairs.idx_t1.to_numpy()][m])}; latent_rows.append(row)
    pd.DataFrame(latent_rows).to_csv(a.output_dir / "latent_distance_metrics.csv", index=False)
    reps = {"z_t": z[pairs.idx_t.to_numpy()], "z_hat_t1": learned, "z_t1": z[pairs.idx_t1.to_numpy()]}
    prog_cols = [f"primary_future_cancer_{h}" for h in HORIZONS]
    run_probe(a, pairs, reps, "prognostic", prog_cols, "prognostic_per_seed_metrics.csv", device)
    diag_reps = {"z_t": reps["z_t"], "z_hat_t1": reps["z_hat_t1"], "z_t1": reps["z_t1"]}
    diag_cols = [f"diag_t_{x}" for x in DIAGNOSTIC_TASKS]
    # The diagnostic-state comparison needs t, predicted t+1, and actual t+1;
    # labels are attached to each state, so run each state against its own label.
    rows = []
    for state, X in diag_reps.items():
        labels = [f"diag_t_{x}" if state == "z_t" else f"diag_t1_{x}" for x in DIAGNOSTIC_TASKS]
        rows_state = run_probe_collect(a, pairs, {state: X}, state, labels, device)
        rows.extend(rows_state)
    pd.DataFrame(rows).to_csv(a.output_dir / "diagnostic_state_per_seed_metrics.csv", index=False)
    if wandb is not None and not a.no_wandb:
        run = wandb.init(project=a.wandb_project, group=a.wandb_group, name="summary", config={"seeds": a.seeds}, reinit="finish_previous", settings=wandb.Settings(root_dir=str(a.output_dir / "wandb"), x_disable_stats=True)); run.log({"latent/test/learned_mse": float(np.mean([r["mse"] for r in latent_rows if r["scope"] == "pooled" and r["representation"] == "learned"]))}); run.finish()


def run_probe_collect(a, pairs, reps, prefix, label_cols, device):
    # Adapter around run_probe that returns rows for one state while retaining
    # the exact same train-only scaling and simple linear probe procedure.
    temp = a.output_dir / f".tmp_{prefix}.csv"; run_probe(a, pairs, reps, prefix, label_cols, temp.name, device)
    d = pd.read_csv(temp); temp.unlink(missing_ok=True); return d.to_dict("records")


if __name__ == "__main__":
    main()
