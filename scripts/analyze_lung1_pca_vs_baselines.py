#!/usr/bin/env python3
"""Frozen PCA16 and PCA32 readouts against Stage+Volume and radiomics with all training labels.

Secondary sensitivity analysis, added after the PCA16 comparisons. Metric: mean outer-fold Uno C at
36 months (repetition 0). Paired patient bootstrap within fixed outer folds, 1,000 resamples, seed
20261026; two-sided Wald z with the bootstrap SE; Holm within each PCA setting (10 tests).
Writes outputs/lung1_nested/pca_vs_baselines.json.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, cohort, locked_folds
from test_lung1_bf16_selected_stats import holm
B, SEED = 1000, 20261026
FMS = ["colipri", "tangerine", "radjepa", "ctclip", "dinov3"]
data = cohort(COHORT); folds = locked_folds(data, FOLDS); rng = np.random.default_rng(SEED)
order, counts, dev = {}, {}, {}
for f in range(5):
    ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == f), "patient_id"])
    test, d = data[data.patient_id.isin(ids)], data[~data.patient_id.isin(ids)]
    order[f] = test.patient_id.tolist()
    s = rng.integers(0, len(test), size=(B, len(test)))
    counts[f] = np.stack([np.bincount(r, minlength=len(test)) for r in s])
    dev[f] = (d.os_time_days.to_numpy(float), d.os_event.to_numpy(int))
def score(frame):
    pts, bs = [], []
    for f in range(5):
        rows = frame[frame.fold == f].set_index("patient_id")
        assert sorted(rows.index) == sorted(order[f])
        n, dd = weighted_pair_matrices(rows.loc[order[f]].reset_index().to_dict("records"), *dev[f])
        pts.append(n.sum() / dd.sum()); bs.append(bootstrap_scores(n, dd, counts[f]))
    return float(np.mean(pts)), np.mean(bs, axis=0)
res = {}
base = pd.read_csv("outputs/lung1_e3_baselines/out_of_fold_predictions.csv", dtype={"patient_id": str})
for b in ["stage_volume", "radiomics"]:
    res[b] = score(base[(base.model == b) & (base.budget_pct == 100)])
for pca in ["pca16", "pca32"]:
    fr = pd.read_csv(f"outputs/lung1_nested/linear_{pca}/out_of_fold_predictions.csv", dtype={"patient_id": str})
    fr = fr[fr.repetition == 0]
    for m in FMS:
        res[(m, pca)] = score(fr[fr.model == m])
out = {}
for pca in ["pca16", "pca32"]:
    rows = []
    for m in FMS:
        for b in ["stage_volume", "radiomics"]:
            d = res[(m, pca)][0] - res[b][0]; dr = res[(m, pca)][1] - res[b][1]; se = np.std(dr, ddof=1)
            rows.append([m, b, res[(m, pca)][0], d, np.percentile(dr, 2.5), np.percentile(dr, 97.5), 2 * stats.norm.sf(abs(d / se)), 2.8016 * se])
    out[pca] = []
    for r, ph in zip(rows, holm([r[6] for r in rows])):
        out[pca].append(dict(zip(["model", "baseline", "uno_c_36m", "difference", "lower", "upper", "p", "mdd"], map(lambda v: v if isinstance(v, str) else float(v), r)), p_holm=float(ph)))
        print(pca, f"{r[0]:9s} {r[1]:12s} C={r[2]:.3f} diff={r[3]:+.3f} [{r[4]:+.3f},{r[5]:+.3f}] p={r[6]:.3f} pHolm={ph:.3f} MDD={r[7]:.3f}")
standalone = {pca: {m: [res[(m, pca)][0], float(np.percentile(res[(m, pca)][1], 2.5)), float(np.percentile(res[(m, pca)][1], 97.5))]
                     for m in FMS} for pca in ["pca16", "pca32"]}  # estimate, 95% interval
json.dump({"bootstrap": B, "seed": SEED, "baselines": {b: res[b][0] for b in ["stage_volume", "radiomics"]},
           "standalone": standalone, **out},
          open("outputs/lung1_nested/pca_vs_baselines.json", "w"), indent=2)
