#!/usr/bin/env python3
"""Paired comparisons of the trainable LUNG1 models against the frozen linear readout.

Protocol: docs/lung1_sweep_validation_protocol.md (amendment 2). Metric: mean over the five
outer folds (repetition 0) of test Uno C at 36 months, censoring weights from each fold's
development set. Paired patient bootstrap within fixed outer folds, 1,000 resamples, seed
20261026; two-sided Wald z with the bootstrap standard error.

Families (Holm within each):
  primary    each MLP, SQA, LoRA and partial fine-tuning model against PCA16 (22 tests)
  vs_pca32   the same contrasts against PCA32 (secondary)
  pca32      PCA32 against PCA16 for each representation (secondary)
For COLIPRI and TANGERINE the LoRA and partial fits are the fixed-epoch sweep runs whose
configuration was chosen on these test folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, cohort, locked_folds
from test_lung1_bf16_selected_stats import holm

ROOT = Path(__file__).resolve().parents[1]
NESTED = ROOT / "outputs/lung1_nested"
SEED = 20261026
FMS = ["colipri", "tangerine", "radjepa", "ctclip", "dinov3"]
TUMOUR = ["pai", "hosny"]
LABELS = {"pca16": "Frozen linear Cox, PCA16", "pca32": "Frozen linear Cox, PCA32",
          "mlp": "Multilayer perceptron", "sqa": "Survival query adapter",
          "lora": "LoRA", "partial": "Partial fine-tuning"}


def source(model: str, method: str) -> tuple[Path, str | None]:
    """Out-of-fold prediction file and, for the shared frozen files, the model filter."""
    if method in ("pca16", "pca32"):
        return NESTED / f"linear_{method}/out_of_fold_predictions.csv", model
    if method in ("mlp", "sqa"):
        return NESTED / f"heads/final/{method}/{model}/out_of_fold_predictions.csv", None
    setting = "lr1e-3_ep20" if method == "lora" else "lr1e-4_ep20"
    root = "lung1_phase3" if model in ("colipri", "tangerine") else "lung1_adaptation_fixed"
    return ROOT / f"outputs/{root}/{setting}/{method}/{model}/out_of_fold_predictions.csv", None


def methods(model: str) -> list[str]:
    return ["mlp", "sqa", "lora", "partial"] if model in FMS else ["mlp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    rng = np.random.default_rng(SEED)
    tests_order, counts, development = {}, {}, {}
    for fold in range(5):
        ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test, dev = data.loc[data.patient_id.isin(ids)], data.loc[~data.patient_id.isin(ids)]
        tests_order[fold] = test.patient_id.tolist()
        sampled = rng.integers(0, len(test), size=(args.bootstrap, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(float), dev.os_event.to_numpy(int))

    point, boot, folds_c = {}, {}, {}
    for model in FMS + TUMOUR:
        for method in ["pca16", "pca32"] + methods(model):
            path, keep = source(model, method)
            frame = pd.read_csv(path, dtype={"patient_id": str})
            if keep is not None:
                frame = frame[frame.model == keep]
            if "repetition" in frame:
                frame = frame[frame.repetition == 0]
            per_fold_point, per_fold_boot = [], []
            for fold in range(5):
                rows = frame[frame.fold == fold].set_index("patient_id")
                if sorted(rows.index) != sorted(tests_order[fold]):
                    raise ValueError(f"{model}/{method}: test patients differ in fold {fold}")
                predictions = rows.loc[tests_order[fold]].reset_index().to_dict(orient="records")
                numerator, denominator = weighted_pair_matrices(predictions, *development[fold])
                per_fold_point.append(numerator.sum() / denominator.sum())
                per_fold_boot.append(bootstrap_scores(numerator, denominator, counts[fold]))
            key = (model, method)
            folds_c[key] = per_fold_point
            point[key] = float(np.mean(per_fold_point))
            boot[key] = np.mean(per_fold_boot, axis=0)

    def contrasts(pairs):
        rows = []
        for first, second in pairs:
            draws = boot[first] - boot[second]
            difference = point[first] - point[second]
            se = float(np.std(draws, ddof=1))
            rows.append({"model": first[0], "method": first[1], "reference": second[1],
                         "difference": difference, "se": se,
                         "lower": float(np.percentile(draws, 2.5)),
                         "upper": float(np.percentile(draws, 97.5)),
                         "p": float(2 * stats.norm.sf(abs(difference / se)))})
        for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
            row["p_holm"] = adjusted
        return rows

    families = {
        "primary": contrasts([((m, k), (m, "pca16")) for m in FMS + TUMOUR for k in methods(m)]),
        "vs_pca32": contrasts([((m, k), (m, "pca32")) for m in FMS + TUMOUR for k in methods(m)]),
        "pca32": contrasts([((m, "pca32"), (m, "pca16")) for m in FMS + TUMOUR]),
    }
    if len(families["primary"]) != 22:
        raise RuntimeError("expected 22 primary comparisons")

    # Keyed by representation and table label for manuscript/scripts/build_results_figures.py.
    result: dict = {m: {} for m in FMS + TUMOUR}
    for row in families["primary"] + families["pca32"]:
        result[row["model"]][LABELS[row["method"]]] = {k: row[k] for k in ("difference", "lower", "upper", "p", "p_holm")}
    result["_meta"] = {
        "metric": "mean outer-fold Uno C at 36 months, repetition 0",
        "bootstrap": args.bootstrap, "seed": SEED,
        "p_values": "two-sided Wald z with bootstrap SE; Holm within each family",
        "note": "conditional on the fitted fold models; COLIPRI and TANGERINE LoRA/partial chosen on these test folds",
        "uno_c_36m": {f"{m}/{k}": {"mean_fold": point[(m, k)], "folds": folds_c[(m, k)]} for m, k in point},
        "families": families,
    }
    out = NESTED / "trainable_tests.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for row in families["primary"]:
        print(f"{row['model']:9s} {row['method']:8s} {point[(row['model'], row['method'])]:.3f} "
              f"diff {row['difference']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}] "
              f"p {row['p']:.3f} holm {row['p_holm']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
