#!/usr/bin/env python3
"""E3: tests of every frozen FM against the two literature baselines.

Same estimator, resamples and seed as `analyze_lung1_e3_baselines.py`: mean of
five outer-fold Uno C through 36 months, averaged over the ten subset draws
below 100%, with 300 paired held-out patient resamples per fold shared by all
models. p-values follow `test_lung1_bf16_selected_stats.py`: two-sided Wald,
z = difference / bootstrap SE. Holm is applied within each family of the 10
contrasts (5 FMs x 2 baselines) at one budget and PCA setting.

Primary family: 100% budget with 16 PCA components. PCA16 was first run as a
post hoc sensitivity in E3, so this choice is declared after the results were
known. All other budget/PCA families are exploratory.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from analyze_lung1_e3_baselines import FM_OUTPUT, FM_PCA16_OUTPUT, OUTPUT, bootstrap_model
from run_lung1_e3_frozen_linear import BUDGETS
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, cohort, locked_folds
from test_lung1_bf16_selected_stats import holm

FMS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
BASELINES = ("stage_volume", "radiomics")
SEED = 20261010


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    rng = np.random.default_rng(SEED)
    counts, development = {}, {}
    for fold in range(5):
        ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test = data.loc[data.patient_id.isin(ids)]
        dev = data.loc[~data.patient_id.isin(ids)]
        sampled = rng.integers(0, len(test), size=(args.bootstrap, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(float), dev.os_event.to_numpy(int))
    names = list(BASELINES) + list(FMS) + [f"{m}_pca16" for m in FMS]
    boot = {name: bootstrap_model(name, OUTPUT, development, counts) for name in names}
    point = {}
    for table, suffix in ((pd.read_csv(OUTPUT / "summary.csv"), ""),
                          (pd.read_csv(FM_OUTPUT / "summary.csv"), ""),
                          (pd.read_csv(FM_PCA16_OUTPUT / "summary.csv"), "_pca16")):
        for row in table.itertuples():
            point[(f"{row.model}{suffix}", int(row.budget_pct))] = float(row.uno_c_36m)
    rows = []
    for pca, suffix in ((16, "_pca16"), (8, "")):
        for budget in BUDGETS:
            family = []
            for fm in FMS:
                for baseline in BASELINES:
                    difference = point[(f"{fm}{suffix}", budget)] - point[(baseline, budget)]
                    draws = boot[f"{fm}{suffix}"][budget] - boot[baseline][budget]
                    se = float(np.std(draws, ddof=1))
                    z = difference / se
                    family.append({"pca_components": pca, "budget_pct": budget, "fm": fm,
                                   "baseline": baseline, "fm_uno_c_36m": point[(f"{fm}{suffix}", budget)],
                                   "baseline_uno_c_36m": point[(baseline, budget)],
                                   "difference": difference, "se": se, "z": z,
                                   "ci95_lower": float(np.percentile(draws, 2.5)),
                                   "ci95_upper": float(np.percentile(draws, 97.5)),
                                   "p": float(2 * stats.norm.sf(abs(z)))})
            for row, adjusted in zip(family, holm([r["p"] for r in family])):
                row["p_holm_family"] = adjusted
                row["family"] = ("primary" if (pca, budget) == (16, 100) else "exploratory")
            rows.extend(family)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "statistical_tests_fm_vs_baselines.csv", index=False)
    (OUTPUT / "statistical_tests_fm_vs_baselines.json").write_text(json.dumps({
        "metric": "mean_outer_fold_uno_c_36m (averaged over 10 subset draws below 100%)",
        "bootstrap": args.bootstrap, "seed": SEED,
        "p_values": "two-sided Wald z with bootstrap SE; Holm within each 10-contrast family (budget x PCA)",
        "primary_family": "budget 100%, PCA 16 (declared after E3 results; PCA16 was a post hoc sensitivity)",
        "scope": "conditional on fitted models, locked folds and fixed subset draws; no refitting",
        "rows": rows}, indent=2) + "\n")
    show = table.assign(ci=lambda t: t.apply(lambda r: f"[{r.ci95_lower:+.3f}, {r.ci95_upper:+.3f}]", axis=1))
    print(show[["pca_components", "budget_pct", "fm", "baseline", "difference", "ci", "p", "p_holm_family"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
