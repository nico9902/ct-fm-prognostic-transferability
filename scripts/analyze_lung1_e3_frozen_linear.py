#!/usr/bin/env python3
"""Summarize E3 learning curves with paired, conditional patient bootstraps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_lung1_e3_frozen_linear import BUDGETS, MODELS, N_DRAWS, OUTPUT
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, HORIZONS, cohort, censor_survival, locked_folds


def weighted_pair_matrices(predictions: list[dict], train_time, train_event):
    t = np.array([item["os_time_days"] for item in predictions], dtype=float)
    e = np.array([item["os_event"] for item in predictions], dtype=int)
    risk = np.array([item["risk"] for item in predictions], dtype=float)
    cases = (e == 1) & (t <= HORIZONS[-1])
    comparable = (t[:, None] < t[None, :]) | ((t[:, None] == t[None, :]) & (e[None, :] == 0))
    g = censor_survival(train_time, train_event)(t)
    if np.any(comparable & cases[:, None] & (g[:, None] <= 0.05)):
        raise ValueError("Insufficient censoring support in E3 bootstrap")
    denominator = comparable * cases[:, None] / np.maximum(g[:, None], 1e-12) ** 2
    difference = risk[:, None] - risk[None, :]
    concordant = (difference > 1e-8).astype(float)
    concordant += 0.5 * (np.abs(difference) <= 1e-8)
    return denominator * concordant, denominator


def bootstrap_scores(numerator, denominator, counts):
    top = np.einsum("bi,ij,bj->b", counts, numerator, counts, optimize=True)
    bottom = np.einsum("bi,ij,bj->b", counts, denominator, counts, optimize=True)
    return np.divide(top, bottom, out=np.full_like(top, np.nan), where=bottom > 0)


def analyze(output: Path, n_boot: int) -> None:
    if not (output / "COMPLETE").exists():
        raise RuntimeError("E3 fits are incomplete")
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    rng = np.random.default_rng(20261010)
    counts = {}
    development = {}
    for fold in range(5):
        test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test = data.loc[data.patient_id.isin(test_ids)]
        dev = data.loc[~data.patient_id.isin(test_ids)]
        sampled = rng.integers(0, len(test), size=(n_boot, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(), dev.os_event.to_numpy())
    scored = {}
    for model in MODELS:
        for budget in BUDGETS:
            by_draw = []
            for draw in range(1 if budget == 100 else N_DRAWS):
                by_fold = []
                for fold in range(5):
                    path = output / "fits" / model / f"fold_{fold}" / f"draw_{draw}_budget_{budget}.json"
                    result = json.loads(path.read_text())
                    numerator, denominator = weighted_pair_matrices(
                        result["predictions"], *development[fold],
                    )
                    boot = bootstrap_scores(numerator, denominator, counts[fold])
                    if not np.isfinite(boot).all():
                        raise ValueError(f"Undefined bootstrap score: {model}, {budget}%, fold {fold}")
                    by_fold.append(boot)
                by_draw.append(np.mean(by_fold, axis=0))
            scored[(model, budget)] = np.mean(by_draw, axis=0)
    summary = pd.read_csv(output / "summary.csv")
    components = json.loads((output / "protocol.json").read_text())["pca_components_fixed"]
    uncertainty = {}
    for model in MODELS:
        uncertainty[model] = {}
        for budget in BUDGETS:
            values = scored[(model, budget)]
            uncertainty[model][str(budget)] = {
                "uno_c_36m_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            }
    contrasts = {}
    for model in MODELS:
        diff = scored[(model, 10)] - scored[(model, 100)]
        contrasts[f"{model}_10_minus_100"] = {
            "estimate": float(summary.loc[(summary.model == model) & (summary.budget_pct == 10), "uno_c_36m"].iloc[0]
                              - summary.loc[(summary.model == model) & (summary.budget_pct == 100), "uno_c_36m"].iloc[0]),
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        }
    interactions = {}
    for index, first in enumerate(MODELS):
        for second in MODELS[index + 1:]:
            first_gain = scored[(first, 100)] - scored[(first, 10)]
            second_gain = scored[(second, 100)] - scored[(second, 10)]
            first_point = summary.loc[summary.model == first].set_index("budget_pct").uno_c_36m
            second_point = summary.loc[summary.model == second].set_index("budget_pct").uno_c_36m
            interactions[f"{first}_minus_{second}_gain"] = {
                "estimate": float((first_point[100] - first_point[10])
                                  - (second_point[100] - second_point[10])),
                "ci95": np.quantile(first_gain - second_gain, [0.025, 0.975]).tolist(),
            }
    payload = {
        "bootstrap_resamples": n_boot,
        "resampling": "test patients resampled within each locked outer fold; shared across all models, budgets, and subset draws",
        "scope": "conditional on fitted models, outer folds, and ten fixed subset draws; models are not refitted",
        "model_budget": uncertainty, "low_minus_full": contrasts,
        "between_model_gain_differences": interactions,
    }
    (output / "paired_uno_bootstrap.json").write_text(json.dumps(payload, indent=2) + "\n")

    colors = {"colipri": "#2b6cb0", "tangerine": "#d97706",
              "radjepa": "#15803d", "ctclip": "#7c3aed", "dinov3": "#0f766e",
              "pai": "#be123c", "hosny": "#57534e"}
    names = {"colipri": "COLIPRI", "tangerine": "TANGERINE",
             "radjepa": "Rad-JEPA-3D", "ctclip": "CT-CLIP", "dinov3": "DINOv3",
             "pai": "Pai (tumour FM)", "hosny": "Hosny (tumour CNN)"}
    fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
    for model in MODELS:
        rows = summary[summary.model == model].sort_values("budget_pct")
        x = rows.events_train_full.to_numpy()
        y = rows.uno_c_36m.to_numpy()
        limits = np.array([uncertainty[model][str(int(b))]["uno_c_36m_ci95"]
                           for b in rows.budget_pct])
        ax.plot(x, y, marker="o", linewidth=2, label=names[model], color=colors[model])
        ax.fill_between(x, limits[:, 0], limits[:, 1], alpha=0.12, color=colors[model])
    ax.axhline(0.5, color="0.55", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([30, 75, 150, 295], labels=["30", "75", "150", "295"])
    ax.set_xlabel("Mean observed deaths in model-fitting subset")
    ax.set_ylabel("Held-out Uno C through 36 months")
    ax.set_title(f"LUNG1 frozen-representation label efficiency ({components} PCs)")
    ax.set_ylim(0.4, 0.78)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output / "event_efficiency_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    analyze(args.output_dir, args.bootstrap)


if __name__ == "__main__":
    main()
