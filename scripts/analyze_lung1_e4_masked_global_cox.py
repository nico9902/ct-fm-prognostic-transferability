#!/usr/bin/env python3
"""Paired conditional comparisons of original, GTV-filled and sham-filled CTs."""

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
from run_lung1_e4_masked_global_cox import EMBEDDINGS, EMBEDDINGS_INPAINT, EMBEDDINGS_BLACK, OUTPUT
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, cohort, locked_folds


def analyze(output: Path, n_boot: int) -> None:
    if not (output / "COMPLETE").exists():
        raise ValueError("Masked-volume Cox fits are incomplete")
    protocol = json.loads((output / "protocol.json").read_text())
    conditions = tuple(protocol["conditions"])
    models = tuple(protocol["models"])
    modified = conditions[1]
    control_condition = conditions[2] if len(conditions) == 3 else None
    contrasts = (("original", modified),)
    extraction = {None: EMBEDDINGS, "inpaint": EMBEDDINGS_INPAINT,
                  "black": EMBEDDINGS_BLACK}[protocol.get("fill_method")]
    full_data = cohort(COHORT)
    folds = locked_folds(full_data, FOLDS)
    if protocol["control_complete_case"]:
        control = pd.read_csv("outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv",
                              dtype={"patient_id": str})
        usable = set(control.loc[control.sham_dx_voxels.notna(), "patient_id"])
        data = full_data.loc[full_data.patient_id.isin(usable)].reset_index(drop=True)
        contrasts += (("original", control_condition), (control_condition, modified))
    else:
        data = full_data
    rng = np.random.default_rng(20261013)
    counts, development = {}, {}
    for fold in range(5):
        ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test = data.loc[data.patient_id.isin(ids)]
        dev = data.loc[~data.patient_id.isin(ids)]
        sampled = rng.integers(0, len(test), size=(n_boot, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(), dev.os_event.to_numpy())
    scores = {}
    for model in models:
        for condition in conditions:
            folds_scores = []
            for fold in range(5):
                fit = json.loads((output / "fits" / model / condition / f"fold_{fold}.json").read_text())
                numerator, denominator = weighted_pair_matrices(
                    fit["predictions"], *development[fold])
                boot = bootstrap_scores(numerator, denominator, counts[fold])
                if not np.isfinite(boot).all():
                    raise ValueError(f"Undefined bootstrap: {model}/{condition}/fold_{fold}")
                folds_scores.append(boot)
            scores[(model, condition)] = np.mean(folds_scores, axis=0)
    summary = pd.read_csv(output / "summary.csv").set_index(["model", "region"])
    rows, intervals = [], []
    for model in models:
        for condition in conditions:
            values = scores[(model, condition)]
            intervals.append({"model": model, "condition": condition,
                              "ci95_lower": float(np.quantile(values, 0.025)),
                              "ci95_upper": float(np.quantile(values, 0.975))})
        for first, second in contrasts:
            values = scores[(model, first)] - scores[(model, second)]
            rows.append({"model": model, "contrast": f"{first}_minus_{second}",
                         "uno_c_36m_difference": float(
                             summary.loc[(model, first), "uno_c_36m"]
                             - summary.loc[(model, second), "uno_c_36m"]),
                         "ci95_lower": float(np.quantile(values, 0.025)),
                         "ci95_upper": float(np.quantile(values, 0.975))})
    pd.DataFrame(rows).to_csv(output / "paired_mask_differences.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "conditional_uno_intervals.csv", index=False)
    if not protocol["control_complete_case"]:
        e3 = pd.read_csv("outputs/lung1_e3_frozen_linear_pca16/summary.csv")
        e3 = e3.loc[e3.budget_pct.eq(100)].set_index("model")
        baseline = [{"model": model, "e3_global_uno_c_36m": e3.loc[model, "uno_c_36m"],
                     "reencoded_original_uno_c_36m": summary.loc[(model, "original"), "uno_c_36m"],
                     "difference": summary.loc[(model, "original"), "uno_c_36m"]
                                   - e3.loc[model, "uno_c_36m"]} for model in models]
        pd.DataFrame(baseline).to_csv(output / "e3_baseline_reencoding_check.csv", index=False)
    input_rows = []
    embedding_rows = []
    for model in models:
        changes = pd.read_csv(extraction / model / "input_change_manifest.csv")
        tumor = changes["gtv_input.mean_abs_input_change"].to_numpy(dtype=float)
        sham = changes["sham_input.mean_abs_input_change"].dropna().to_numpy(dtype=float)
        ratios = (changes["gtv_input.mean_abs_input_change"] /
                  changes["sham_input.mean_abs_input_change"].clip(lower=1e-8)).dropna()
        input_rows.append({"model": model, "n_patients": len(changes),
                           "n_zero_gtv_input_masks": int((changes["gtv_input.mask_voxels"] == 0).sum()),
                           "n_valid_sham_masks": int(changes["sham_input.mask_voxels"].notna().sum()),
                           "n_zero_sham_input_masks": int((changes["sham_input.mask_voxels"] == 0).sum()),
                           "median_gtv_input_voxels": float(changes["gtv_input.mask_voxels"].median()),
                           "median_sham_input_voxels": float(changes["sham_input.mask_voxels"].median()),
                           "median_gtv_mean_abs_change": float(np.median(tumor)),
                           "median_sham_mean_abs_change": float(np.median(sham)),
                           "median_gtv_to_sham_change_ratio": float(ratios.median())})
        for condition in conditions[1:]:
            relative_l2, cosine = [], []
            for pid in data.patient_id:
                original = np.load(EMBEDDINGS / model / "original" / f"{pid}.npy")
                altered = np.load(extraction / model / condition / f"{pid}.npy")
                denominator = max(float(np.linalg.norm(original)), 1e-12)
                relative_l2.append(float(np.linalg.norm(altered - original) / denominator))
                cosine.append(float(np.dot(original, altered) /
                                    (denominator * max(float(np.linalg.norm(altered)), 1e-12))))
            embedding_rows.append({"model": model, "condition": condition,
                                   "n_patients": len(relative_l2),
                                   "median_relative_l2": float(np.median(relative_l2)),
                                   "p95_relative_l2": float(np.quantile(relative_l2, 0.95)),
                                   "median_cosine": float(np.median(cosine))})
    pd.DataFrame(input_rows).to_csv(output / "input_change_summary.csv", index=False)
    pd.DataFrame(embedding_rows).to_csv(output / "embedding_change_summary.csv", index=False)
    (output / "bootstrap_protocol.json").write_text(json.dumps({
        "resamples": n_boot, "seed": 20261013,
        "method": "paired held-out patient resampling within each locked outer fold; fixed development censoring curves",
        "scope": "conditional on fitted models, fixed folds and cohort; no refitting or multiplicity adjustment"
    }, indent=2) + "\n")
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 4.2),
                             sharey=True, constrained_layout=True)
    labels = {"original": "Original", "gtv_filled": "GTV filled", "sham_filled": "Control filled",
              "gtv_inpaint": "GTV inpaint", "sham_inpaint": "Control inpaint",
              "gtv_black": "GTV black", "sham_black": "Control black"}
    for axis, model in zip(np.atleast_1d(axes), models):
        x = np.arange(len(conditions))
        center = np.array([summary.loc[(model, condition), "uno_c_36m"] for condition in conditions])
        lower = np.array([np.quantile(scores[(model, condition)], 0.025) for condition in conditions])
        upper = np.array([np.quantile(scores[(model, condition)], 0.975) for condition in conditions])
        axis.errorbar(x, center, yerr=np.vstack([center - lower, upper - center]),
                      fmt="o", capsize=3, color="#255a89")
        axis.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
        axis.set_xticks(x, [labels[c] for c in conditions], rotation=45, ha="right")
        axis.set_title(model.upper())
        axis.grid(axis="y", alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("Held-out Uno C through 36 months")
    fig.savefig(output / "masked_volume_uno_c.png", dpi=180)
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
