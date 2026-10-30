#!/usr/bin/env python3
"""Paired conditional uncertainty and regional contrast plot for E4 Cox probes."""

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
from run_lung1_e4_regional_cox import ALL_MODELS, MODELS, OUTPUT, REGIONS
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, cohort, locked_folds


CONTRASTS = (("tumor", "all"), ("non_gtv", "all"),
             ("tumor", "non_gtv"), ("peri20", "tumor"),
             ("tumor_peri20", "tumor"))


def analyze(output: Path, n_boot: int, merge_dirs: tuple[Path, ...] = (),
            write_dir: Path | None = None) -> None:
    """Analyze one fit directory, optionally merged with others sharing cohort and folds."""
    sources = (output, *merge_dirs)
    model_dir, summaries = {}, []
    for source in sources:
        if not (source / "COMPLETE").exists():
            raise ValueError(f"E4 regional Cox fits are incomplete: {source}")
        protocol = json.loads((source / "protocol.json").read_text())
        locked = {k: v for k, v in protocol.items() if k not in (
            "models", "token_manifests_sha256", "gtv_mapping_sha256", "peri20_mapping_sha256",
            "gtv_patient_table_sha256", "peri20_patient_table_sha256")}
        if source == output:
            reference = locked
        elif locked != reference:
            raise ValueError(f"{source}: cohort, folds, probe or exclusions differ")
        for model in protocol.get("models", MODELS):
            model_dir.setdefault(model, source)
        summaries.append(pd.read_csv(source / "summary.csv"))
    models = tuple(m for m in ALL_MODELS if m in model_dir)
    write_dir = write_dir or output
    write_dir.mkdir(parents=True, exist_ok=True)
    full_data = cohort(COHORT)
    folds = locked_folds(full_data, FOLDS)
    exclusions = json.loads((output / "protocol.json").read_text()).get("excluded_patient_ids", [])
    data = full_data.loc[~full_data.patient_id.isin(exclusions)].reset_index(drop=True)
    rng = np.random.default_rng(20261012)
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
        for region in REGIONS:
            by_fold = []
            for fold in range(5):
                fit = json.loads((model_dir[model] / "fits" / model / region / f"fold_{fold}.json").read_text())
                numerator, denominator = weighted_pair_matrices(
                    fit["predictions"], *development[fold])
                boot = bootstrap_scores(numerator, denominator, counts[fold])
                if not np.isfinite(boot).all():
                    raise ValueError(f"Undefined bootstrap: {model}/{region}/fold_{fold}")
                by_fold.append(boot)
            scores[(model, region)] = np.mean(by_fold, axis=0)
    summary = pd.concat(summaries).drop_duplicates(["model", "region"]).set_index(["model", "region"])
    rows = []
    for model in models:
        for first, second in CONTRASTS:
            difference = scores[(model, first)] - scores[(model, second)]
            rows.append({"model": model, "contrast": f"{first}_minus_{second}",
                         "uno_c_36m_difference": float(summary.loc[(model, first), "uno_c_36m"]
                                                       - summary.loc[(model, second), "uno_c_36m"]),
                         "ci95_lower": float(np.quantile(difference, 0.025)),
                         "ci95_upper": float(np.quantile(difference, 0.975))})
    pd.DataFrame(rows).to_csv(write_dir / "paired_regional_differences.csv", index=False)
    interactions = []
    for first_model_index, first_model in enumerate(models):
        for second_model in models[first_model_index + 1:]:
            for first_region, second_region in CONTRASTS:
                first_delta = scores[(first_model, first_region)] - scores[(first_model, second_region)]
                second_delta = scores[(second_model, first_region)] - scores[(second_model, second_region)]
                interaction = first_delta - second_delta
                point = ((summary.loc[(first_model, first_region), "uno_c_36m"]
                          - summary.loc[(first_model, second_region), "uno_c_36m"])
                         - (summary.loc[(second_model, first_region), "uno_c_36m"]
                            - summary.loc[(second_model, second_region), "uno_c_36m"]))
                interactions.append({"first_model": first_model, "second_model": second_model,
                                     "regional_contrast": f"{first_region}_minus_{second_region}",
                                     "difference_of_differences": float(point),
                                     "ci95_lower": float(np.quantile(interaction, 0.025)),
                                     "ci95_upper": float(np.quantile(interaction, 0.975))})
    pd.DataFrame(interactions).to_csv(write_dir / "between_model_regional_interactions.csv",
                                      index=False)
    ci = [{"model": model, "region": region,
           "ci95_lower": float(np.quantile(scores[(model, region)], 0.025)),
           "ci95_upper": float(np.quantile(scores[(model, region)], 0.975))}
          for model in models for region in REGIONS]
    pd.DataFrame(ci).to_csv(write_dir / "conditional_uno_intervals.csv", index=False)
    (write_dir / "bootstrap_protocol.json").write_text(json.dumps({
        "resamples": n_boot, "seed": 20261012,
        "method": "paired held-out patient resampling within each locked outer fold; fixed development censoring curves",
        "scope": "conditional on fitted models, token maps, fixed folds and cohort; no model refitting or multiplicity adjustment",
        "fit_directories": {model: str(path) for model, path in model_dir.items()}
    }, indent=2) + "\n")
    labels = {"all": "All", "tumor": "Tumor", "non_gtv": "Non-GTV",
              "peri20": "0–20 mm", "tumor_peri20": "Tumor + 0–20 mm"}
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 4.3), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        positions = np.arange(len(REGIONS))
        center = np.array([summary.loc[(model, region), "uno_c_36m"] for region in REGIONS])
        lower = np.array([np.quantile(scores[(model, region)], 0.025) for region in REGIONS])
        upper = np.array([np.quantile(scores[(model, region)], 0.975) for region in REGIONS])
        axis.errorbar(positions, center, yerr=np.vstack([center - lower, upper - center]),
                      fmt="o", capsize=3, color="#255a89")
        axis.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
        axis.set_xticks(positions, [labels[region] for region in REGIONS], rotation=55, ha="right")
        axis.set_title({"dinov3": "DINOv3"}.get(model, model.upper()))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Held-out Uno C through 36 months")
    fig.savefig(write_dir / "regional_uno_c.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--merge-dir", type=Path, action="append", default=[],
                        help="Further fit directory with the same cohort, folds and probe")
    parser.add_argument("--write-dir", type=Path,
                        help="Where to write merged results (default: --output-dir)")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    analyze(args.output_dir, args.bootstrap, tuple(args.merge_dir), args.write_dir)


if __name__ == "__main__":
    main()
