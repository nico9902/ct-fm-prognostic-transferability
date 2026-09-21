#!/usr/bin/env python3
"""Does the ACTUAL future frozen COLIPRI representation carry more prognostic
information about lung cancer than the current representation?

This is a purely observational, retrospective probing experiment on frozen
features. No world-model / next-state predictor is trained here -- z_t1 is
always the COLIPRI embedding of the real, observed next scan. It establishes
the upper-bound motivation for a future predictive-latent stage: if even the
*actual* future state is not more decodable than the present state, learning
to predict that future state cannot help either.

Inputs (all reused, nothing is re-extracted or re-derived from raw DICOM):
  - Frozen COLIPRI mean-pooled embeddings already cached for the diagnostic
    decodability cohort (999 patients / 2984 scans):
    outputs/colipri_decodability/mean/cache/{embeddings.npy,embedding_metadata.csv}
    Mean pooling is used because it was the best-performing arm (highest
    AUROC/AUPRC/MCC on all four diagnostic targets) among mean/max/attention
    in outputs/colipri_decodability/completed_report.md.
  - Scan-level prognostic labels built previously:
    data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv
  - The existing patient-level train/val/test split used for the diagnostic
    decodability study: outputs/colipri_decodability/patient_splits.csv
    (599/200/200 patients). Reused verbatim so every representation
    condition below is compared on identical patients/pairs/splits.

Pair construction: for every patient, a T0->T1 pair requires both T0 and T1
embeddings; a T1->T2 pair requires both T1 and T2. Both pair types are
pooled into one dataset (a patient's two pairs, if both exist, always land
in the same split, since the split is assigned per patient -- no leakage).

Endpoint definitions (see docs/nlst_prognostic_labels_report.md for how the
underlying per-scan cancer_event/cancer_time_days/future_cancer_*y columns
are computed):
  - PRIMARY: the target is the *t+1 scan's own* landmark-safe label, i.e.
    "cancer within Xy after t+1", already excluding pairs where the first
    cancer was confirmed on or before t+1 (prognostic_label_quality_flags on
    the t+1 row contains excluded_prior_cancer_diagnosis in that case, and
    the label is NA). This is the fair, leakage-safe comparison.
  - SECONDARY (explicitly disadvantages nothing and is reported separately,
    clearly flagged): the target is the *t scan's own* label, i.e. "cancer
    within Xy after t". Because t1 sits strictly after t, z_t1 may already
    reflect a diagnosis-adjacent state for patients diagnosed between t and
    t1, so a z_t1 advantage here is expected and not evidence of predicting
    something in z_t1's own future.

Representation conditions (same frozen 864-d mean-pooled vector; no new
encoder training): z_t, z_t1, concat(z_t, z_t1) [1728-d], delta_z = z_t1-z_t.
Same linear probe architecture/training recipe as the diagnostic
decodability study (DiagnosticLightningModule = single Linear(D,1), BCE with
training-only class weight, training-only StandardScaler, AdamW, early
stopping on val loss), 5 seeds.

A simple linear Cox proportional-hazards model (time-to-cancer after t+1) is
also fit per representation condition. Given train-set sizes (~1150-1190)
are far smaller than the raw feature dimension (864/1728), a full-dimension
penalized Cox fit is both numerically unstable and prohibitively slow
(single-fit wall time measured at 45s for d=864 and >180s for d=1728, before
any penalizer search); PCA (fit on train only, after the same train-only
StandardScaler) reduces to 32 components first. This PCA step is specific to
the Cox model; the classification linear probes above always use the full
864/1728/864-d space, matching the task's request.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_diagnostic_state_decodability import DiagnosticLightningModule, metrics, seed_all
from run_tangerine_decodability import METRICS

HORIZONS = ["1y", "2y", "3y"]
CONDITIONS = ["z_t", "z_t1", "concat", "delta_z"]
ANALYSES = ["primary", "secondary"]
SEEDS = list(range(5))
PENALIZER_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
PCA_COMPONENTS = 32
NUMERIC_COLS = ["cancer_event", "cancer_time_days", "future_cancer_1y", "future_cancer_2y", "future_cancer_3y"]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-cache", type=Path, default=Path("outputs/colipri_decodability/mean/cache"))
    p.add_argument("--prognostic-labels", type=Path,
                    default=Path("data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv"))
    p.add_argument("--patient-splits", type=Path, default=Path("outputs/colipri_decodability/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/colipri_future_state_prognosis"))
    p.add_argument("--device", default="cuda:3")
    return p.parse_args()


def build_pairs(a: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    z = np.load(a.embedding_cache / "embeddings.npy")
    em = pd.read_csv(a.embedding_cache / "embedding_metadata.csv", dtype=str)
    prog = pd.read_csv(a.prognostic_labels, dtype=str)
    splits = pd.read_csv(a.patient_splits, dtype=str)[["patient_id", "split"]]

    prog = prog.drop(columns=["screening_round"])
    m = em.merge(prog, on=["patient_id", "series_uid"], how="left", validate="one_to_one")
    for c in NUMERIC_COLS:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["embedding_index"] = m["embedding_index"].astype(int)

    fields = ["embedding_index", "prognostic_label_quality_flags"] + NUMERIC_COLS
    wide = m.set_index(["patient_id", "screening_round"])[fields].unstack("screening_round")

    def make(t: str, t1: str) -> pd.DataFrame:
        idx_t, idx_t1 = wide[("embedding_index", t)], wide[("embedding_index", t1)]
        avail = idx_t.notna() & idx_t1.notna()
        pids = wide.index[avail]
        rows = {
            "patient_id": pids, "pair_type": f"{t}_{t1}",
            "idx_t": idx_t[avail].astype(int).to_numpy(), "idx_t1": idx_t1[avail].astype(int).to_numpy(),
            "flags_t": wide[("prognostic_label_quality_flags", t)][avail].to_numpy(),
            "flags_t1": wide[("prognostic_label_quality_flags", t1)][avail].to_numpy(),
        }
        for c in NUMERIC_COLS:
            rows[f"primary_{c}"] = wide[(c, t1)][avail].to_numpy()
            rows[f"secondary_{c}"] = wide[(c, t)][avail].to_numpy()
        return pd.DataFrame(rows)

    pairs = pd.concat([make("T0", "T1"), make("T1", "T2")], ignore_index=True)
    pairs = pairs.merge(splits, on="patient_id", how="left", validate="many_to_one")
    assert pairs["split"].notna().all(), "every embedded patient must be in the reused patient split"
    return pairs, z


def representation(z: np.ndarray, idx_t: np.ndarray, idx_t1: np.ndarray, condition: str) -> np.ndarray:
    zt, zt1 = z[idx_t], z[idx_t1]
    if condition == "z_t":
        return zt
    if condition == "z_t1":
        return zt1
    if condition == "concat":
        return np.concatenate([zt, zt1], axis=1)
    if condition == "delta_z":
        return zt1 - zt
    raise ValueError(condition)


def train_probe(xtr, ytr, xva, yva, xte, yte, seed: int, device: str):
    seed_all(seed)
    dim = xtr.shape[1]
    pos_weight = float((1 - ytr).sum() / ytr.sum())
    module = DiagnosticLightningModule(dim, pos_weight)
    sets = [TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) for x, y in
            [(xtr, ytr), (xva, yva), (xte, yte)]]
    train = DataLoader(sets[0], batch_size=min(128, len(sets[0])), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val = DataLoader(sets[1], batch_size=min(128, len(sets[1])))
    trainer = pl.Trainer(
        accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=[int(device.split(":")[-1])] if device.startswith("cuda") else 1,
        max_epochs=100, logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=15, min_delta=1e-5)],
    )
    trainer.fit(module, train, val)
    module.eval()
    with torch.inference_mode():
        prob = torch.sigmoid(module(torch.from_numpy(xte).to(module.device))).cpu().numpy()
    result = metrics(yte, prob)
    result["epochs_run"] = int(trainer.current_epoch)
    return result, prob


def run_classification(a: argparse.Namespace, pairs: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    preds_dir = a.output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    path = a.output_dir / "per_seed_results.csv"
    if path.exists():
        rows = pd.read_csv(path).to_dict("records")

    for analysis in ANALYSES:
        for horizon in HORIZONS:
            col = f"{analysis}_future_cancer_{horizon}"
            mask = pairs[col].notna().to_numpy()
            sub = pairs[mask].reset_index(drop=True)
            y_all = sub[col].to_numpy(dtype=np.float32)
            tr, va, te = [sub["split"].eq(s).to_numpy() for s in ["train", "val", "test"]]
            for condition in CONDITIONS:
                if any(r["analysis"] == analysis and r["horizon"] == horizon and r["condition"] == condition
                       and r.get("status") == "skipped_no_positive_class" for r in rows):
                    continue
                if y_all[tr].sum() == 0 or y_all[te].sum() == 0:
                    rows.append({"analysis": analysis, "horizon": horizon, "condition": condition,
                                 "status": "skipped_no_positive_class", "n_train": int(tr.sum()),
                                 "n_val": int(va.sum()), "n_test": int(te.sum())})
                    pd.DataFrame(rows).to_csv(path, index=False)
                    continue
                X_all = representation(z, sub["idx_t"].to_numpy(), sub["idx_t1"].to_numpy(), condition).astype(np.float32)
                scaler = StandardScaler().fit(X_all[tr])
                Xs = scaler.transform(X_all).astype(np.float32)
                for seed in SEEDS:
                    if any(r["analysis"] == analysis and r["horizon"] == horizon and r["condition"] == condition
                           and r.get("seed") == seed for r in rows):
                        continue
                    result, prob = train_probe(Xs[tr], y_all[tr], Xs[va], y_all[va], Xs[te], y_all[te], seed, a.device)
                    rows.append({"analysis": analysis, "horizon": horizon, "condition": condition, "seed": seed,
                                 "status": "ok", "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                                 "n_test_positive": int(y_all[te].sum()), **result})
                    pd.DataFrame(rows).to_csv(path, index=False)
                    pred_df = sub.loc[te, ["patient_id", "pair_type"]].copy()
                    pred_df["label"] = y_all[te]
                    pred_df["probability"] = prob
                    pred_df.to_csv(preds_dir / f"{analysis}_{horizon}_{condition}_seed{seed}.csv", index=False)
                    print(analysis, horizon, condition, seed, {k: result[k] for k in ["auroc", "auprc", "mcc", "f1"]}, flush=True)

    out = pd.DataFrame(rows)
    ok = out[out.status.eq("ok")]
    if len(ok):
        agg = ok.groupby(["analysis", "horizon", "condition"])[METRICS].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join(c).strip("_") for c in agg.columns]
        n_info = ok.groupby(["analysis", "horizon", "condition"])[["n_train", "n_val", "n_test", "n_test_positive"]].first().reset_index()
        agg = agg.merge(n_info, on=["analysis", "horizon", "condition"])
        agg.to_csv(a.output_dir / "summary_results.csv", index=False)
    return out


def run_survival(a: argparse.Namespace, pairs: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    mask = pairs["primary_cancer_event"].notna().to_numpy()
    sub = pairs[mask].reset_index(drop=True)
    T = sub["primary_cancer_time_days"].to_numpy(dtype=float)
    E = sub["primary_cancer_event"].to_numpy(dtype=int)
    tr, va, te = [sub["split"].eq(s).to_numpy() for s in ["train", "val", "test"]]

    rows = []
    for condition in CONDITIONS:
        X = representation(z, sub["idx_t"].to_numpy(), sub["idx_t1"].to_numpy(), condition).astype(np.float32)
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X)
        n_comp = min(PCA_COMPONENTS, int(tr.sum()) - 1)
        pca = PCA(n_components=n_comp, random_state=0).fit(Xs[tr])
        Xp = pca.transform(Xs)
        cols = [f"pc{i}" for i in range(Xp.shape[1])]

        def frame(m):
            d = pd.DataFrame(Xp[m], columns=cols)
            d["T"] = T[m]
            d["E"] = E[m]
            return d

        train_df, val_df, test_df = frame(tr), frame(va), frame(te)

        def c_index(cph, df):
            risk = cph.predict_partial_hazard(df)
            return concordance_index(df["T"], -risk, df["E"])

        best = None
        for pen in PENALIZER_GRID:
            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train_df, duration_col="T", event_col="E")
                val_c = c_index(cph, val_df)
            except Exception as exc:  # non-convergence at this penalizer
                print(f"survival {condition} penalizer={pen} failed: {exc}", flush=True)
                continue
            if best is None or val_c > best[1]:
                best = (pen, val_c, cph)
        if best is None:
            rows.append({"condition": condition, "status": "failed_to_converge",
                         "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())})
            continue
        pen, val_c, cph = best
        test_c = c_index(cph, test_df)
        rows.append({"condition": condition, "status": "ok", "penalizer": pen, "pca_components": n_comp,
                     "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                     "n_events_train": int(E[tr].sum()), "n_events_val": int(E[va].sum()), "n_events_test": int(E[te].sum()),
                     "val_c_index": val_c, "test_c_index": test_c})
        print("survival", condition, rows[-1], flush=True)
    return pd.DataFrame(rows)


def comparison_tables(a: argparse.Namespace, classification: pd.DataFrame):
    ok = classification[classification.status.eq("ok")]
    if not len(ok):
        return
    agg = ok.groupby(["analysis", "horizon", "condition"])[METRICS].agg(["mean", "std"])
    for metric in ["auroc", "auprc", "mcc", "f1"]:
        rows = []
        for analysis in ANALYSES:
            for horizon in HORIZONS:
                row = {"Analysis": analysis, "Horizon": horizon}
                for condition in CONDITIONS:
                    key = (analysis, horizon, condition)
                    if key in agg.index:
                        m, s = agg.loc[key, (metric, "mean")], agg.loc[key, (metric, "std")]
                        row[condition] = f"{m:.3f} ± {s:.3f}"
                    else:
                        row[condition] = "n/a"
                rows.append(row)
        pd.DataFrame(rows).to_csv(a.output_dir / f"comparison_{metric}.csv", index=False)


def qc_stats(pairs: pd.DataFrame) -> dict:
    stats = {"n_pairs_total": len(pairs), "n_pairs_by_type": pairs.pair_type.value_counts().to_dict(),
             "n_patients": int(pairs.patient_id.nunique()), "n_pairs_by_split": pairs.split.value_counts().to_dict()}
    for analysis in ANALYSES:
        elig = pairs[f"{analysis}_cancer_event"].notna()
        stats[f"{analysis}_endpoint"] = {
            "n_eligible": int(elig.sum()),
            "n_events": int((pairs[f"{analysis}_cancer_event"] == 1).sum()),
            "n_censored": int((pairs[f"{analysis}_cancer_event"] == 0).sum()),
        }
        for horizon in HORIZONS:
            col = f"{analysis}_future_cancer_{horizon}"
            v = pairs[col]
            n_valid = int(v.notna().sum())
            stats[f"{analysis}_endpoint"][f"future_cancer_{horizon}"] = {
                "n_labeled": n_valid, "n_positive": int((v == 1).sum()), "n_negative": int((v == 0).sum()),
                "prevalence": float((v == 1).sum() / n_valid) if n_valid else None,
            }
    return stats


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    pairs, z = build_pairs(a)
    pairs.to_csv(a.output_dir / "pairs_manifest.csv", index=False)
    (a.output_dir / "qc.json").write_text(json.dumps(qc_stats(pairs), indent=2, default=str))
    print(json.dumps(qc_stats(pairs), indent=2, default=str), flush=True)

    classification = run_classification(a, pairs, z)
    comparison_tables(a, classification)

    survival = run_survival(a, pairs, z)
    survival.to_csv(a.output_dir / "survival_results.csv", index=False)


if __name__ == "__main__":
    main()
