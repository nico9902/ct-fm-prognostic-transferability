#!/usr/bin/env python3
"""Plot matched E3 baselines and estimate paired Uno C differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_e3_baselines import BASELINES, OUTPUT
from run_lung1_e3_frozen_linear import BUDGETS, N_DRAWS
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, ROOT, cohort, locked_folds


FM_OUTPUT = ROOT / "outputs/lung1_e3_frozen_linear"
FM_PCA16_OUTPUT = ROOT / "outputs/lung1_e3_frozen_linear_pca16"
PAIRED_MODELS = ("stage_volume", "radiomics", "colipri", "colipri_pca16", "dinov3", "dinov3_pca16")
REFERENCE_FMS = ("colipri", "dinov3")


def source(model: str, baseline_output: Path) -> tuple[Path, str]:
    if model in BASELINES:
        return baseline_output, model
    if model.endswith("_pca16"):
        return FM_PCA16_OUTPUT, model.removesuffix("_pca16")
    return FM_OUTPUT, model


def bootstrap_model(model: str, output: Path, development: dict, counts: dict) -> dict:
    folder, label = source(model, output)
    result = {}
    for budget in BUDGETS:
        draws = []
        for draw in range(1 if budget == 100 else N_DRAWS):
            folds = []
            for fold in range(5):
                path = folder / "fits" / label / f"fold_{fold}" / f"draw_{draw}_budget_{budget}.json"
                fit = json.loads(path.read_text())
                numerator, denominator = weighted_pair_matrices(
                    fit["predictions"], *development[fold])
                folds.append(bootstrap_scores(numerator, denominator, counts[fold]))
            draws.append(np.mean(folds, axis=0))
        result[budget] = np.mean(draws, axis=0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if not (args.output_dir / "COMPLETE").exists():
        raise RuntimeError("Baseline fits are incomplete")
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    rng = np.random.default_rng(20261010)
    counts, development = {}, {}
    for fold in range(5):
        ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test = data.loc[data.patient_id.isin(ids)]
        dev = data.loc[~data.patient_id.isin(ids)]
        sampled = rng.integers(0, len(test), size=(args.bootstrap, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(float), dev.os_event.to_numpy(int))
    boot = {model: bootstrap_model(model, args.output_dir, development, counts)
            for model in PAIRED_MODELS}
    base = pd.read_csv(args.output_dir / "summary.csv")
    fm = pd.read_csv(FM_OUTPUT / "summary.csv")
    fm16 = pd.read_csv(FM_PCA16_OUTPUT / "summary.csv")
    all_rows = pd.concat([base, fm16], ignore_index=True)
    names = {"stage_volume": "Stage + tumor volume", "radiomics": "GTV radiomics",
             "colipri": "COLIPRI (PCA16)", "tangerine": "TANGERINE (PCA16)",
             "radjepa": "Rad-JEPA-3D (PCA16)", "ctclip": "CT-CLIP (PCA16)",
             "dinov3": "DINOv3 (PCA16)"}
    colors = {"stage_volume": "#111827", "radiomics": "#be123c",
              "colipri": "#2b6cb0", "tangerine": "#d97706",
              "radjepa": "#15803d", "ctclip": "#7c3aed", "dinov3": "#0f766e"}
    fig, ax = plt.subplots(figsize=(8.3, 5.3), constrained_layout=True)
    for model in names:
        rows = all_rows.loc[all_rows.model == model].sort_values("budget_pct")
        x = rows.events_train_full.to_numpy()
        y = rows.uno_c_36m.to_numpy()
        style = "-" if model in BASELINES else "--"
        width = 2.4 if model in BASELINES else 1.6
        ax.plot(x, y, marker="o", linestyle=style, linewidth=width, color=colors[model],
                label=names[model])
        if model in BASELINES:
            intervals = np.array([np.quantile(boot[model][int(b)], [0.025, 0.975])
                                  for b in rows.budget_pct])
            ax.fill_between(x, intervals[:, 0], intervals[:, 1], color=colors[model], alpha=0.1)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([29, 74, 148, 295], labels=["29", "74", "148", "295"])
    ax.set_xlabel("Mean observed deaths in model-fitting subset")
    ax.set_ylabel("Held-out Uno C through 36 months")
    ax.set_title("E3: matched Cox curves with 16-component FM probes")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(args.output_dir / "matched_learning_curves.png", dpi=180)
    plt.close(fig)
    for reference in REFERENCE_FMS:
        contrasts = []
        for baseline in BASELINES:
            for target in (reference, f"{reference}_pca16"):
                target_table = fm16 if target.endswith("_pca16") else fm
                for budget in BUDGETS:
                    a = base.loc[(base.model == baseline) & (base.budget_pct == budget), "uno_c_36m"].iloc[0]
                    b = target_table.loc[(target_table.model == reference) &
                                         (target_table.budget_pct == budget), "uno_c_36m"].iloc[0]
                    interval = np.quantile(boot[baseline][budget] - boot[target][budget],
                                           [0.025, 0.975])
                    contrasts.append({"baseline": baseline, "reference": target,
                                      "budget_pct": budget, "delta_uno_c_36m": float(a - b),
                                      "ci95_lower": float(interval[0]), "ci95_upper": float(interval[1])})
        pd.DataFrame(contrasts).to_csv(args.output_dir / f"paired_{reference}_differences.csv", index=False)
    baseline_differences = []
    for budget in BUDGETS:
        radiomics = base.loc[(base.model == "radiomics") &
                              (base.budget_pct == budget), "uno_c_36m"].iloc[0]
        clinical = base.loc[(base.model == "stage_volume") &
                             (base.budget_pct == budget), "uno_c_36m"].iloc[0]
        interval = np.quantile(boot["radiomics"][budget] - boot["stage_volume"][budget],
                               [0.025, 0.975])
        baseline_differences.append({"budget_pct": budget,
                                     "radiomics_minus_stage_volume": float(radiomics - clinical),
                                     "ci95_lower": float(interval[0]),
                                     "ci95_upper": float(interval[1])})
    pd.DataFrame(baseline_differences).to_csv(
        args.output_dir / "paired_baseline_differences.csv", index=False)
    payload = {model: {str(b): np.quantile(value, [0.025, 0.975]).tolist()
                       for b, value in per_budget.items()} for model, per_budget in boot.items()}
    (args.output_dir / "bootstrap_intervals.json").write_text(json.dumps({
        "resamples": args.bootstrap,
        "scope": "conditional on locked folds, fitted models, and subset draws; no model refitting",
        "intervals": payload}, indent=2) + "\n")


if __name__ == "__main__":
    main()
