#!/usr/bin/env python3
"""Statistical tests for the selected-checkpoint five-fold LUNG1 comparison.

Covers all 25 method-encoder combinations (five encoders including DINOv3)
and the two literature baselines. Uses the estimator, draws and seed of
`run_lung1_bf16_selected_baselines.py`: pooled Uno C at 36 months on the 416
out-of-fold predictions, ranked by -log S(36 m), 500 paired patient bootstrap
resamples with the censoring curve fixed to the full cohort.

Tests (Wald z / chi-square on the bootstrap covariance; Holm within family):
  chance      pooled Uno C versus 0.5, per condition
  omnibus     equality of the five encoders within each rung (4 df)
  encoders    all 10 encoder pairs within each rung
  baselines   each condition versus stage + volume and versus radiomics
  rungs       each adapted rung versus the frozen linear rung of its encoder
  friedman    Friedman test of fold-level Uno C across encoders within a rung
Also writes the five-fold mean Harrell and full-follow-up Uno C of the linear rung.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_lung1_bf16_selected_baselines import (  # noqa: E402
    BOOTSTRAP_SEED, COHORT, FM_METHODS, FM_MODELS, OUTPUT, SELECTED, pooled_risk,
)
from run_lung1_e2_encoder_bf16_lightning import write_json  # noqa: E402
from run_lung1_e3_baselines import BASELINES  # noqa: E402
from run_lung1_frozen_cox_e1 import cohort, uno_c  # noqa: E402

FULL_FOLLOW_UP = 3653.0


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def wald(difference: float, boot: np.ndarray) -> dict:
    se = float(np.nanstd(boot, ddof=1))
    z = difference / se
    return {"difference": difference, "se": se, "z": z,
            "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
            "p": float(2 * stats.norm.sf(abs(z)))}


def family(tests: dict) -> dict:
    keys = list(tests)
    for key, p in zip(keys, holm([tests[k]["p"] for k in keys])):
        tests[key]["p_holm"] = p
    return tests


def fold_c_indices(data: pd.DataFrame, method: str, model: str) -> pd.DataFrame:
    frame = pd.read_csv(SELECTED / method / model / "out_of_fold_predictions.csv",
                        dtype={"patient_id": str})
    rows = []
    for fold, test in frame.groupby("fold"):
        train = data[~data.patient_id.isin(test.patient_id)]
        t, e, r = (test.os_time_days.to_numpy(float), test.os_event.to_numpy(int),
                   test.risk.to_numpy(float))
        rows.append({"fold": fold,
                     "harrell_c": concordance_index(t, -r, e),
                     "uno_c_full": uno_c(train.os_time_days.to_numpy(float),
                                         train.os_event.to_numpy(int), t, e, r, tau=FULL_FOLLOW_UP)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    data = cohort(COHORT)
    t = data.os_time_days.to_numpy(dtype=float)
    e = data.os_event.to_numpy(dtype=int)
    risks = {f"baseline/{name}": pooled_risk(OUTPUT / name / "out_of_fold_predictions.csv", data)
             for name in BASELINES}
    for method in FM_METHODS:
        for model in FM_MODELS:
            risks[f"{method}/{model}"] = pooled_risk(
                SELECTED / method / model / "out_of_fold_predictions.csv", data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(data), len(data)) for _ in range(args.bootstrap)]
    point = {key: float(uno_c(t, e, t, e, value)) for key, value in risks.items()}
    boot = {key: np.array([uno_c(t, e, t[d], e[d], value[d]) for d in draws])
            for key, value in risks.items()}

    chance = family({key: wald(point[key] - 0.5, boot[key]) for key in risks})
    encoders, baselines, rungs, omnibus, friedman = {}, {}, {}, {}, {}
    for method in FM_METHODS:
        keys = [f"{method}/{model}" for model in FM_MODELS]
        for a, b in itertools.combinations(keys, 2):
            encoders[f"{a} - {b}"] = wald(point[a] - point[b], boot[a] - boot[b])
        for key in keys:
            for name in BASELINES:
                other = f"baseline/{name}"
                baselines[f"{key} - {other}"] = wald(point[key] - point[other], boot[key] - boot[other])
        # Equality of all encoders: contrasts against the first, chi-square on bootstrap covariance.
        diffs = np.array([point[k] - point[keys[0]] for k in keys[1:]])
        cov = np.cov(np.stack([boot[k] - boot[keys[0]] for k in keys[1:]]))
        chi2 = float(diffs @ np.linalg.solve(cov, diffs))
        omnibus[method] = {"chi2": chi2, "df": len(diffs), "p": float(stats.chi2.sf(chi2, len(diffs)))}
        folds = np.stack([pd.read_csv(SELECTED / method / model / "fold_metrics.csv")
                          .sort_values("fold").uno_c_36m.to_numpy() for model in FM_MODELS])
        statistic, p = stats.friedmanchisquare(*folds)
        friedman[method] = {"statistic": float(statistic), "df": len(FM_MODELS) - 1, "p": float(p),
                            "mean_rank": dict(zip(FM_MODELS, stats.rankdata(-folds, axis=0)
                                                  .mean(axis=1).tolist()))}
    for model in FM_MODELS:
        base = f"linear/{model}"
        for method in FM_METHODS[1:]:
            key = f"{method}/{model}"
            rungs[f"{key} - {base}"] = wald(point[key] - point[base], boot[key] - boot[base])
    family(omnibus)
    family(friedman)

    linear_c = {}
    for model in FM_MODELS:
        frame = fold_c_indices(data, "linear", model)
        linear_c[model] = {"harrell_c_mean_fold": float(frame.harrell_c.mean()),
                           "uno_c_full_mean_fold": float(frame.uno_c_full.mean()),
                           "folds": frame.to_dict(orient="records")}

    result = {"metric": "pooled_uno_c_36m", "risk": "negative_log_predicted_survival_at_36_months",
              "bootstrap": args.bootstrap, "seed": BOOTSTRAP_SEED,
              "p_values": "two-sided Wald z with bootstrap standard error; Holm within each family",
              "note": "conditional on fitted fold models and the locked split",
              "conditions": {key: {"uno_c_36m": point[key],
                                   "ci95": [float(np.nanpercentile(boot[key], 2.5)),
                                            float(np.nanpercentile(boot[key], 97.5))]}
                             for key in risks},
              "chance": chance, "omnibus_encoders": omnibus, "friedman_folds": friedman,
              "encoders": family(encoders), "baselines": family(baselines), "rungs": family(rungs),
              "linear_full_follow_up": linear_c}
    write_json(SELECTED / "statistical_tests.json", result)
    for name in ("omnibus_encoders", "friedman_folds"):
        for key, value in result[name].items():
            print(f"{name} {key}: p={value['p']:.4f} holm={value['p_holm']:.4f}")


if __name__ == "__main__":
    main()
