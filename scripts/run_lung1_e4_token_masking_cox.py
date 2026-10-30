#!/usr/bin/env python3
"""E4 token masking: matched linear Cox probes and paired comparisons.

Each feature set (readout x condition) is refitted separately on the locked E3
outer folds with training-only standardization, fixed PCA, and ridge Cox, like
the other E4 probes. Paired intervals resample held-out patients within each
fixed fold, shared across feature sets and models.

Readouts
* keep:   mean of final native tokens at positions retained in every condition
          (outside GTV tokens on 416 patients; outside GTV and control tokens on 403).
* global: each encoder's official global readout over its retained tokens.
* all:    mean of all final native tokens, original forward (416 only); compared
          with keep/original it shows the effect of dropping tumor *positions*
          without hiding them inside the encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_lung1_e4_regional_cox as regional
from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_e3_frozen_linear import write_json
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, ROOT, cohort, locked_folds, sha256


MODELS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
NAMES = {"colipri": "COLIPRI", "tangerine": "TANGERINE", "radjepa": "Rad-JEPA-3D",
         "ctclip": "CT-CLIP", "dinov3": "DINOv3"}
EMBEDDINGS = ROOT / "outputs/lung1_e4_token_masking/embeddings"
CONTROLS = ROOT / "outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv"
SEED = 20261013


def feature_sets(control: bool) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(readout, condition) pairs to fit, and (first, second) feature contrasts."""
    conditions = ("original", "gtv_masked", "control_masked") if control else ("original", "gtv_masked")
    keep = "keep_both" if control else "keep_gtv"
    features = [(readout, c) for readout in (keep, "global") for c in conditions]
    contrasts = []
    for readout in (keep, "global"):
        contrasts.append(((readout, "original"), (readout, "gtv_masked")))
        if control:
            contrasts += [((readout, "original"), (readout, "control_masked")),
                          ((readout, "control_masked"), (readout, "gtv_masked"))]
    if not control:
        features.append(("all", "original"))
        contrasts.append((("all", "original"), (keep, "original")))
    return features, contrasts


def label(feature: tuple[str, str]) -> str:
    return f"{feature[0]}__{feature[1]}"


def load_matrices(model: str, data: pd.DataFrame, features) -> dict[str, np.ndarray]:
    folder = EMBEDDINGS / model
    if not (folder / "COMPLETE").exists():
        raise ValueError(f"{model}: token-masked extraction incomplete")
    result = {label(f): [] for f in features}
    for pid in data.patient_id:
        arrays = np.load(folder / "vectors" / f"{pid}.npz")
        for readout, condition in features:
            result[label((readout, condition))].append(arrays[f"{condition}__{readout}"])
    return {key: np.stack(value).astype(np.float32) for key, value in result.items()}


def analyze(output: Path, data: pd.DataFrame, folds: pd.DataFrame, features, contrasts,
            models, n_boot: int) -> None:
    rng = np.random.default_rng(SEED)
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
        for feature in features:
            per_fold = []
            for fold in range(5):
                fit = json.loads((output / "fits" / model / label(feature) / f"fold_{fold}.json").read_text())
                numerator, denominator = weighted_pair_matrices(fit["predictions"], *development[fold])
                boot = bootstrap_scores(numerator, denominator, counts[fold])
                if not np.isfinite(boot).all():
                    raise ValueError(f"Undefined bootstrap: {model}/{label(feature)}/fold_{fold}")
                per_fold.append(boot)
            scores[(model, label(feature))] = np.mean(per_fold, axis=0)
    summary = pd.read_csv(output / "summary.csv").set_index(["model", "region"])
    intervals, differences = [], []
    for model in models:
        for feature in features:
            values = scores[(model, label(feature))]
            intervals.append({"model": model, "feature": label(feature),
                              "uno_c_36m": summary.loc[(model, label(feature)), "uno_c_36m"],
                              "harrell_c_full": summary.loc[(model, label(feature)), "harrell_c_full"],
                              "ci95_lower": float(np.quantile(values, 0.025)),
                              "ci95_upper": float(np.quantile(values, 0.975))})
        for first, second in contrasts:
            values = scores[(model, label(first))] - scores[(model, label(second))]
            differences.append({"model": model, "contrast": f"{label(first)}_minus_{label(second)}",
                                "uno_c_36m_difference": float(
                                    summary.loc[(model, label(first)), "uno_c_36m"]
                                    - summary.loc[(model, label(second)), "uno_c_36m"]),
                                "ci95_lower": float(np.quantile(values, 0.025)),
                                "ci95_upper": float(np.quantile(values, 0.975))})
    pd.DataFrame(intervals).to_csv(output / "conditional_uno_intervals.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "paired_mask_differences.csv", index=False)
    changes = []
    for model in models:
        matrices = load_matrices(model, data, features)
        for readout, condition in features:
            if condition == "original":
                continue
            base = matrices[label((readout, "original"))]
            altered = matrices[label((readout, condition))]
            rel = np.linalg.norm(altered - base, axis=1) / np.maximum(np.linalg.norm(base, axis=1), 1e-12)
            changes.append({"model": model, "feature": label((readout, condition)),
                            "median_relative_l2_vs_original": float(np.median(rel)),
                            "p95_relative_l2_vs_original": float(np.quantile(rel, 0.95))})
    pd.DataFrame(changes).to_csv(output / "embedding_change_summary.csv", index=False)
    write_json(output / "bootstrap_protocol.json", {
        "resamples": n_boot, "seed": SEED,
        "method": "paired held-out patient resampling within each locked outer fold; fixed development censoring curves",
        "scope": "conditional on fitted models, fixed folds and cohort; no refitting or multiplicity adjustment"})
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 4.4), sharey=True,
                             constrained_layout=True)
    for axis, model in zip(np.atleast_1d(axes), models):
        x = np.arange(len(features))
        center = np.array([summary.loc[(model, label(f)), "uno_c_36m"] for f in features])
        lower = np.array([np.quantile(scores[(model, label(f))], 0.025) for f in features])
        upper = np.array([np.quantile(scores[(model, label(f))], 0.975) for f in features])
        colors = ["#255a89" if f[1] == "original" else "#b45309" if f[1] == "gtv_masked" else "#6b7280"
                  for f in features]
        axis.scatter(x, center, color=colors, zorder=3)
        axis.vlines(x, lower, upper, color=colors)
        axis.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
        axis.set_xticks(x, [label(f).replace("__", "\n") for f in features], rotation=60, ha="right",
                        fontsize=7)
        axis.set_title(NAMES[model])
        axis.grid(axis="y", alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("Held-out Uno C through 36 months")
    fig.savefig(output / "token_masking_uno_c.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true",
                        help="403 patients with a valid control; adds the control-masked condition")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    regional.COMPONENTS = args.components
    full_data = cohort(COHORT)
    folds = locked_folds(full_data, FOLDS)
    if args.control:
        control = pd.read_csv(CONTROLS, dtype={"patient_id": str})
        usable = set(control.loc[control.sham_dx_voxels.notna(), "patient_id"])
        data = full_data.loc[full_data.patient_id.isin(usable)].reset_index(drop=True)
    else:
        data = full_data
    features, contrasts = feature_sets(args.control)
    name = ("cox_control403" if args.control else "cox_full416") + (
        "" if args.components == 16 else f"_pca{args.components}")
    output = args.output_dir or ROOT / "outputs/lung1_e4_token_masking" / name
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "models": list(MODELS), "features": [label(f) for f in features],
                "extraction_protocols_sha256": {m: sha256(EMBEDDINGS / m / "protocol.json") for m in MODELS},
                "outer_repetition": 0, "outer_folds": 5, "pca_components": args.components,
                "penalties": [0.01, 0.1, 1.0],
                "penalty_selection": "3fold_event_stratified_uno_c_36m_within_outer_development",
                "primary_metric": "mean_outer_fold_uno_c_36m",
                "design": "separate_train_and_test_per_feature_set",
                "included_patient_ids_sha256": hashlib.sha256("\n".join(data.patient_id).encode()).hexdigest(),
                "n_patients": len(data), "control_complete_case": args.control}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Token-masking Cox protocol differs; use a new output directory")
    write_json(path, protocol)
    keys = [(m, label(f), fold) for m in MODELS for f in features for fold in range(5)]
    last_death_day = float(data.loc[data.os_event == 1, "os_time_days"].max())
    for model in MODELS:
        if args.only_model and model != args.only_model:
            continue
        todo = [(f, fold) for f in features for fold in range(5)
                if not (output / "fits" / model / label(f) / f"fold_{fold}.json").exists()]
        if not todo:
            continue
        matrices = load_matrices(model, data, features)
        for feature, fold in todo:
            fit = regional.fit_one(data, matrices[label(feature)], model, label(feature),
                                   fold, folds, last_death_day)
            write_json(output / "fits" / model / label(feature) / f"fold_{fold}.json", fit)
            print(f"{model} {label(feature)} fold={fold} Uno36={fit['metrics']['uno_c_36m']:.3f}", flush=True)
    regional.aggregate(output, keys)
    if (output / "COMPLETE").exists() and not args.fit_only:
        analyze(output, data, folds, features, contrasts, MODELS, args.bootstrap)
        print(pd.read_csv(output / "paired_mask_differences.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
