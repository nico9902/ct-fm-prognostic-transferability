#!/usr/bin/env python3
"""Incremental value of frozen FM embeddings over stage + tumor volume.

Locked E3/E4 split: five outer folds (repetition zero), full development pool,
penalty chosen by the same three event-stratified inner folds and grid.

* clinical:        stage II/IIIa/IIIb/missing indicators + log1p GTV volume
* clinical + FM:   the same covariates plus the FM global embedding reduced to a
                   fixed number of PCA components (training-only scaling and PCA)

Both are fitted with the same lifelines L2 Cox (one penalty for all covariates,
clinical covariates standardized), so a difference comes only from adding the
FM block. FM-only scores are the E3 full-budget fits on the same folds.

Primary test: clinical + FM minus clinical, mean outer-fold Uno C at 36 months,
PCA16, two-sided Wald with paired bootstrap SE, Holm over the five FMs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_e3_baselines import features as baseline_features
from run_lung1_e3_frozen_linear import inner_assignments, write_json
from run_lung1_frozen_cox_e1 import (
    ALPHAS, COHORT, FOLDS, GRID, ROOT, SEED, cohort, fold_metrics, load_embeddings,
    locked_folds, sha256, uno_c,
)
from test_lung1_bf16_selected_stats import holm

FMS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
OUTPUT = ROOT / "outputs/lung1_clinical_plus_fm"
E3 = {16: ROOT / "outputs/lung1_e3_frozen_linear_pca16", 8: ROOT / "outputs/lung1_e3_frozen_linear"}
BOOT_SEED = 20261018


class Design:
    """Training-only transform: standardized clinical block + PCA scores of the FM block."""

    def __init__(self, clinical: np.ndarray, fm: np.ndarray | None, components: int):
        # The stage-missing indicator (one patient) is constant in some training sets.
        self.keep = np.std(clinical, axis=0) > 0
        self.clinical = StandardScaler().fit(clinical[:, self.keep])
        self.fm = None
        if fm is not None:
            self.fm_scaler = StandardScaler().fit(fm)
            self.fm = PCA(n_components=components, svd_solver="randomized",
                          random_state=SEED).fit(self.fm_scaler.transform(fm))

    def __call__(self, clinical: np.ndarray, fm: np.ndarray | None) -> pd.DataFrame:
        blocks = [self.clinical.transform(clinical[:, self.keep])]
        columns = [f"clin_{i}" for i in np.flatnonzero(self.keep)]
        if self.fm is not None:
            blocks.append(self.fm.transform(self.fm_scaler.transform(fm)))
            columns += [f"pc_{i:02d}" for i in range(self.fm.n_components_)]
        return pd.DataFrame(np.hstack(blocks), columns=columns)


CLINICAL_PENALTY = None  # None: one shared penalty; else fixed penalty for clinical columns


def fit(x: pd.DataFrame, time: np.ndarray, event: np.ndarray, alpha: float) -> CoxPHFitter:
    frame = x.assign(time=time, event=event)
    penalty = alpha
    if CLINICAL_PENALTY is not None:
        penalty = np.array([CLINICAL_PENALTY if c.startswith("clin_") else alpha for c in x.columns])
    return CoxPHFitter(penalizer=penalty, l1_ratio=0.0).fit(
        frame, duration_col="time", event_col="event", show_progress=False)


def fit_fold(data, clinical, fm, fold, folds, components, last_death, name):
    test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
    test = data.patient_id.isin(test_ids).to_numpy()
    dev = ~test
    train = data.loc[dev].reset_index(drop=True)
    inner = inner_assignments(train, fold, 0, 100)
    c_dev, c_test = clinical[dev], clinical[test]
    f_dev = fm[dev] if fm is not None else None
    f_test = fm[test] if fm is not None else None
    scores = []
    for alpha in ALPHAS:
        fold_scores = []
        for inner_fold in range(3):
            valid = train.patient_id.isin(inner.loc[inner.inner_fold == inner_fold, "patient_id"]).to_numpy()
            tr = ~valid
            design = Design(c_dev[tr], None if fm is None else f_dev[tr], components)
            model = fit(design(c_dev[tr], None if fm is None else f_dev[tr]),
                        train.os_time_days.to_numpy(float)[tr], train.os_event.to_numpy(int)[tr], alpha)
            risk = model.predict_log_partial_hazard(
                design(c_dev[valid], None if fm is None else f_dev[valid])).to_numpy().reshape(-1)
            score = uno_c(train.os_time_days.to_numpy(float)[tr], train.os_event.to_numpy(int)[tr],
                          train.os_time_days.to_numpy(float)[valid], train.os_event.to_numpy(int)[valid], risk)
            if not np.isfinite(score):
                raise ValueError("Undefined inner Uno C")
            fold_scores.append(score)
        scores.append(float(np.mean(fold_scores)))
    alpha = ALPHAS[int(np.argmax(scores))]
    design = Design(c_dev, f_dev, components)
    time, event = train.os_time_days.to_numpy(float), train.os_event.to_numpy(int)
    model = fit(design(c_dev, f_dev), time, event, alpha)
    x_test = design(c_test, f_test)
    risk = model.predict_log_partial_hazard(x_test).to_numpy().reshape(-1)
    survival = model.predict_survival_function(x_test, times=GRID).to_numpy().T
    held = data.loc[test]
    t_time, t_event = held.os_time_days.to_numpy(float), held.os_event.to_numpy(int)
    metrics = fold_metrics(time, event, t_time, t_event, risk, survival)
    metrics["harrell_c_full"] = float(concordance_index(t_time, -risk, t_event))
    metrics["uno_c_full"] = uno_c(time, event, t_time, t_event, risk, tau=last_death)
    coef = model.params_
    return {"model": name, "fold": fold, "alpha": float(alpha), "inner_uno_c_36m": max(scores),
            "n_train": int(dev.sum()), "n_test": int(test.sum()),
            "clinical_coefficients": {k: float(v) for k, v in coef.items() if k.startswith("clin_")},
            "fm_coefficient_l2": float(np.linalg.norm([v for k, v in coef.items() if k.startswith("pc_")]))
            if fm is not None else 0.0,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "predictions": [{"patient_id": str(p), "os_time_days": float(a), "os_event": int(b), "risk": float(r)}
                            for p, a, b, r in zip(held.patient_id, t_time, t_event, risk)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--clinical-penalty", type=float,
                        help="Sensitivity: fixed L2 penalty for clinical columns; the grid then tunes only the FM block")
    args = parser.parse_args()
    global CLINICAL_PENALTY
    CLINICAL_PENALTY = args.clinical_penalty
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    clinical, clinical_info = baseline_features(data, "stage_volume", Path())
    output = OUTPUT / (f"pca{args.components}" + (
        "" if args.clinical_penalty is None else f"_clinpen{args.clinical_penalty:g}"))
    output.mkdir(parents=True, exist_ok=True)
    fms = {}
    for fm in FMS:
        manifest = ROOT / "outputs/lung1_e1_embeddings" / fm / "original_ct" / "embedding_manifest.csv"
        _, fms[fm], _, digest = load_embeddings(f"{fm}={manifest}", data)
    write_json(output / "protocol.json", {
        "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS), "outer_repetition": 0,
        "clinical": clinical_info["columns"], "fm_pca_components": args.components,
        "penalties": list(ALPHAS), "penalty_selection": "3fold_event_stratified_uno_c_36m_within_outer_development (E3 inner seeds)",
        "cox": "lifelines L2; clinical standardized; FM block standardized then PCA",
        "clinical_penalty": "shared with FM block (selected)" if args.clinical_penalty is None else args.clinical_penalty,
        "primary": "clinical+FM minus clinical, mean outer-fold Uno C 36m, PCA16, Wald with paired bootstrap SE, Holm over 5 FMs",
        "bootstrap": args.bootstrap, "bootstrap_seed": BOOT_SEED})
    last_death = float(data.loc[data.os_event == 1, "os_time_days"].max())
    names = ["clinical"] + [f"clinical+{fm}" for fm in FMS]
    fits = {}
    for name in names:
        fm = None if name == "clinical" else fms[name.split("+")[1]]
        fits[name] = []
        for fold in range(5):
            result = fit_fold(data, clinical, fm, fold, folds, args.components, last_death, name)
            write_json(output / "fits" / name / f"fold_{fold}.json", result)
            fits[name].append(result)
        print(f"{name}: Uno36={np.mean([r['metrics']['uno_c_36m'] for r in fits[name]]):.4f}", flush=True)
    for fm in FMS:  # FM only, from the E3 full-budget fits on the same folds and inner splits
        fits[fm] = [json.loads((E3[args.components] / "fits" / fm / f"fold_{f}" / "draw_0_budget_100.json")
                               .read_text()) for f in range(5)]
    rng = np.random.default_rng(BOOT_SEED)
    counts, development = {}, {}
    for fold in range(5):
        ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test, dev = data.loc[data.patient_id.isin(ids)], data.loc[~data.patient_id.isin(ids)]
        sampled = rng.integers(0, len(test), size=(args.bootstrap, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(float), dev.os_event.to_numpy(int))
    boot, point, summary = {}, {}, []
    for name, results in fits.items():
        per_fold = []
        for fold, result in enumerate(results):
            if [p["patient_id"] for p in result["predictions"]] != [
                    p["patient_id"] for p in fits["clinical"][fold]["predictions"]]:
                raise ValueError(f"{name}: held-out patient order differs in fold {fold}")
            numerator, denominator = weighted_pair_matrices(result["predictions"], *development[fold])
            per_fold.append(bootstrap_scores(numerator, denominator, counts[fold]))
        boot[name] = np.mean(per_fold, axis=0)
        point[name] = float(np.mean([r["metrics"]["uno_c_36m"] for r in results]))
        summary.append({"model": name, "uno_c_36m": point[name],
                        "ci95_lower": float(np.percentile(boot[name], 2.5)),
                        "ci95_upper": float(np.percentile(boot[name], 97.5)),
                        "harrell_c_full": float(np.mean([r["metrics"]["harrell_c_full"] for r in results])),
                        "ibs_3m_36m": float(np.mean([r["metrics"]["ibs_3m_36m"] for r in results])),
                        "alphas": [r.get("alpha") for r in results]})
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    tests = []
    for family, (first, second) in (("added_value_of_fm", ("clinical+{fm}", "clinical")),
                                    ("added_value_of_clinical", ("clinical+{fm}", "{fm}"))):
        rows = []
        for fm in FMS:
            a, b = first.format(fm=fm), second.format(fm=fm)
            draws = boot[a] - boot[b]
            difference = point[a] - point[b]
            se = float(np.std(draws, ddof=1))
            rows.append({"family": family, "fm": fm, "contrast": f"{a} - {b}", "difference": difference,
                         "se": se, "ci95_lower": float(np.percentile(draws, 2.5)),
                         "ci95_upper": float(np.percentile(draws, 97.5)),
                         "p": float(2 * stats.norm.sf(abs(difference / se))),
                         "ibs_difference": next(s["ibs_3m_36m"] for s in summary if s["model"] == a)
                         - next(s["ibs_3m_36m"] for s in summary if s["model"] == b)})
        for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
            row["p_holm"] = adjusted
        tests.extend(rows)
    pd.DataFrame(tests).to_csv(output / "tests.csv", index=False)
    print(pd.DataFrame(summary).drop(columns="alphas").round(4).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    (output / "COMPLETE").write_text("done\n")


if __name__ == "__main__":
    main()
