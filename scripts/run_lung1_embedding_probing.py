#!/usr/bin/env python3
"""Linear probing of clinical variables from frozen global embeddings (phase 1.3).

What do the frozen FM embeddings encode? Each clinical variable is predicted from
each encoder's official global vector, original and with GTV tokens hidden inside
the encoder (E4 token masking vectors), on the locked outer folds of repetition zero.

Probes (fitted on training patients only): standardization, then
* pca16: PCA to 16 components (same capacity as the Cox probes), ridge / L2 logistic
* full:  all dimensions, ridge / L2 logistic
with the L2 strength chosen by 3-fold inner CV. Patients missing a target are dropped
for that target.

Metrics on pooled out-of-fold predictions: Spearman rho and R2 for continuous
targets, ROC AUC for binary targets; 95% patient-bootstrap intervals, and paired
original - masked differences on the same resamples.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_lung1_e3_frozen_linear import write_json
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, ROOT, SEED, cohort, locked_folds, sha256

MODELS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
EMBEDDINGS = ROOT / "outputs/lung1_e4_token_masking/embeddings"
OUTPUT = ROOT / "outputs/lung1_embedding_probing"
CONDITIONS = ("original", "gtv_masked")
BOOT_SEED = 20261019


def targets(data: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    t = pd.to_numeric(data.t_stage, errors="coerce")
    n = pd.to_numeric(data.n_stage, errors="coerce")
    stage = data.overall_stage.astype(str).str.lower()
    histology = data.histology.astype(str).str.lower()
    known_hist = ~histology.isin(("nan", "nos"))
    return {
        "age": ("continuous", pd.to_numeric(data.age, errors="coerce")),
        "log_gtv_volume": ("continuous", np.log1p(pd.to_numeric(data.gtv_volume_cm3, errors="raise"))),
        "male": ("binary", (data.sex.astype(str).str.lower() == "male").astype(float)),
        "T3_T4": ("binary", (t >= 3).astype(float).where(t.notna())),
        "N_positive": ("binary", (n >= 1).astype(float).where(n.notna())),
        "N2_N3": ("binary", (n >= 2).astype(float).where(n.notna())),
        "stage_III": ("binary", stage.str.startswith("iii").astype(float).where(stage != "nan")),
        "squamous": ("binary", (histology == "squamous cell carcinoma").astype(float).where(known_hist)),
        "adenocarcinoma": ("binary", (histology == "adenocarcinoma").astype(float).where(known_hist)),
    }


def probe(kind: str, capacity: str):
    steps = [StandardScaler()]
    if capacity == "pca16":
        steps.append(PCA(n_components=16, svd_solver="randomized", random_state=SEED))
    if kind == "continuous":
        steps.append(RidgeCV(alphas=np.logspace(-3, 5, 17), cv=3))
    else:
        steps.append(LogisticRegressionCV(Cs=np.logspace(-5, 3, 17), cv=3, scoring="roc_auc",
                                          max_iter=5000, random_state=SEED))
    return make_pipeline(*steps)


def metric(kind: str, y: np.ndarray, pred: np.ndarray) -> float:
    if kind == "continuous":
        return float(stats.spearmanr(y, pred).statistic)
    return float(roc_auc_score(y, pred))


def run_one(task: tuple) -> dict:
    model, condition, capacity, target, kind, x, y, fold_of = task
    keep = np.isfinite(y)
    pred = np.full(len(y), np.nan)
    for fold in range(5):
        train = keep & (fold_of != fold)
        test = keep & (fold_of == fold)
        fitted = probe(kind, capacity).fit(x[train], y[train])
        pred[test] = fitted.predict(x[test]) if kind == "continuous" else fitted.predict_proba(x[test])[:, 1]
    return {"model": model, "condition": condition, "capacity": capacity, "target": target,
            "kind": kind, "pred": pred}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    fold_of = data.patient_id.map(folds.loc[folds.repetition == 0].set_index("patient_id").fold).to_numpy(int)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    goals = targets(data)
    write_json(OUTPUT / "protocol.json", {
        "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS), "outer_repetition": 0,
        "embeddings": {m: sha256(EMBEDDINGS / m / "protocol.json") for m in MODELS},
        "readout": "official global vector (E4 token-masking extraction)", "conditions": list(CONDITIONS),
        "capacities": ["pca16", "full"], "targets": {k: v[0] for k, v in goals.items()},
        "inner_selection": "3-fold CV within training (RidgeCV / LogisticRegressionCV roc_auc)",
        "bootstrap": args.bootstrap, "bootstrap_seed": BOOT_SEED})
    tasks = []
    for model in MODELS:
        vectors = {c: [] for c in CONDITIONS}
        for pid in data.patient_id:
            arrays = np.load(EMBEDDINGS / model / "vectors" / f"{pid}.npz")
            for c in CONDITIONS:
                vectors[c].append(arrays[f"{c}__global"])
        for condition in CONDITIONS:
            x = np.stack(vectors[condition]).astype(np.float64)
            for capacity in ("pca16", "full"):
                for target, (kind, y) in goals.items():
                    tasks.append((model, condition, capacity, target, kind, x, y.to_numpy(float), fold_of))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_one, tasks))
    predictions = pd.DataFrame({"patient_id": data.patient_id})
    for r in results:
        predictions[f"{r['model']}|{r['condition']}|{r['capacity']}|{r['target']}"] = r["pred"]
    predictions.to_csv(OUTPUT / "out_of_fold_predictions.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    samples = rng.integers(0, len(data), size=(args.bootstrap, len(data)))
    boot, rows = {}, []
    for r in results:
        y = goals[r["target"]][1].to_numpy(float)
        point = metric(r["kind"], y[np.isfinite(y)], r["pred"][np.isfinite(y)])
        draws = []
        for s in samples:
            ys, ps = y[s], r["pred"][s]
            ok = np.isfinite(ys)
            draws.append(metric(r["kind"], ys[ok], ps[ok]) if r["kind"] == "continuous" or len(np.unique(ys[ok])) == 2
                         else np.nan)
        key = (r["model"], r["condition"], r["capacity"], r["target"])
        boot[key] = np.array(draws)
        row = {"model": r["model"], "condition": r["condition"], "capacity": r["capacity"],
               "target": r["target"], "metric": "spearman" if r["kind"] == "continuous" else "auc",
               "value": point, "ci95_lower": float(np.nanpercentile(draws, 2.5)),
               "ci95_upper": float(np.nanpercentile(draws, 97.5)), "n": int(np.isfinite(y).sum())}
        if r["kind"] == "continuous":
            ok = np.isfinite(y)
            row["r2"] = float(r2_score(y[ok], r["pred"][ok]))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    diffs = []
    for (model, condition, capacity, target), draws in boot.items():
        if condition != "original":
            continue
        masked = boot[(model, "gtv_masked", capacity, target)]
        pick = lambda c: summary.loc[(summary.model == model) & (summary.condition == c)
                                     & (summary.capacity == capacity) & (summary.target == target), "value"].item()
        d = draws - masked
        diffs.append({"model": model, "capacity": capacity, "target": target,
                      "original_minus_masked": pick("original") - pick("gtv_masked"),
                      "ci95_lower": float(np.nanpercentile(d, 2.5)), "ci95_upper": float(np.nanpercentile(d, 97.5))})
    pd.DataFrame(diffs).to_csv(OUTPUT / "paired_masking_differences.csv", index=False)
    table = summary.loc[summary.capacity == "pca16"].pivot_table(
        index=["target", "metric"], columns=["model", "condition"], values="value")
    print(table.round(3).to_string())
    (OUTPUT / "COMPLETE").write_text("done\n")


if __name__ == "__main__":
    main()
