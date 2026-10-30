#!/usr/bin/env python3
"""Paired bootstrap contrasts across the four completed frozen-Cox E1 conditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, HORIZONS, SEED, censor_survival, cohort, locked_folds

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "original_pca32": ROOT / "outputs/lung1_e1_frozen_cox_four_models",
    "original_ridge": ROOT / "outputs/lung1_e1_frozen_cox_original_ridge",
    "crop_pca32": ROOT / "outputs/lung1_e1_frozen_cox_crop_pca32",
    "crop_ridge": ROOT / "outputs/lung1_e1_frozen_cox_crop_ridge",
}
MODELS = ("colipri", "tangerine", "radjepa", "ctclip")
CONTRASTS = {
    "crop_effect_pca32": ("crop_pca32", "original_pca32"),
    "crop_effect_ridge": ("crop_ridge", "original_ridge"),
    "ridge_effect_original": ("original_ridge", "original_pca32"),
    "ridge_effect_crop": ("crop_ridge", "crop_pca32"),
}
BOOTSTRAP = 500
OUTPUT = ROOT / "outputs/lung1_e1_condition_comparison/paired_uno_c.json"


def score(time: np.ndarray, event: np.ndarray, risk: np.ndarray,
          censor_probability: np.ndarray) -> float:
    cases = (event == 1) & (time <= HORIZONS[-1])
    earlier = cases[:, None] & ((time[:, None] < time[None, :]) |
             ((time[:, None] == time[None, :]) & (event[None, :] == 0)))
    if np.any(earlier & (censor_probability[:, None] <= 0.05)):
        return np.nan
    weights = 1.0 / np.maximum(censor_probability, 1e-12) ** 2
    denominator = np.sum(earlier * weights[:, None])
    if denominator == 0:
        return np.nan
    difference = risk[:, None] - risk[None, :]
    comparable = (difference > 1e-8).astype(float)
    comparable += 0.5 * (np.abs(difference) <= 1e-8)
    return float(np.sum(earlier * comparable * weights[:, None]) / denominator)


def main() -> None:
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    protocols = {name: json.loads((path / "protocol.json").read_text()) for name, path in RUNS.items()}
    for field in ("cohort_sha256", "folds_sha256", "n_patients", "n_events"):
        if len({json.dumps(protocol[field]) for protocol in protocols.values()}) != 1:
            raise ValueError(f"The four runs differ at {field}")
    predictions = {name: pd.read_csv(path / "out_of_fold_predictions.csv") for name, path in RUNS.items()}
    patient_index = {patient_id: index for index, patient_id in enumerate(data.patient_id)}
    groups = {name: {(int(rep), int(fold), model): part.sort_values("patient_id").reset_index(drop=True)
                     for (rep, fold, model), part in frame.groupby(["repetition", "fold", "model"])}
              for name, frame in predictions.items()}
    conditions = tuple(RUNS)
    units = []
    for repetition in range(5):
        for fold in range(5):
            test_ids = set(folds.loc[(folds.repetition == repetition) &
                                     (folds.fold == fold), "patient_id"])
            training = data.loc[~data.patient_id.isin(test_ids)]
            g = censor_survival(training.os_time_days.to_numpy(), training.os_event.to_numpy())
            reference = groups[conditions[0]][(repetition, fold, MODELS[0])]
            if len(reference) not in (83, 84):
                raise ValueError(f"Unexpected held-out fold size at {repetition}/{fold}")
            time = reference.os_time_days.to_numpy()
            event = reference.os_event.to_numpy()
            risk = {}
            for condition in conditions:
                for model in MODELS:
                    part = groups[condition][(repetition, fold, model)]
                    if not (part.patient_id.equals(reference.patient_id)
                            and np.array_equal(part.os_time_days.to_numpy(), time)
                            and np.array_equal(part.os_event.to_numpy(), event)):
                        raise ValueError(f"Prediction patient alignment differs: {condition}/{model}/{repetition}/{fold}")
                    risk[(condition, model)] = part.risk.to_numpy()
            indices = np.array([patient_index[patient_id] for patient_id in reference.patient_id])
            units.append((indices, time, event, g(time), risk))
    observed = {key: np.mean([score(time, event, risk[key], probability)
                              for _, time, event, probability, risk in units])
                for key in ((condition, model) for condition in conditions for model in MODELS)}
    for condition, directory in RUNS.items():
        summary = json.loads((directory / "summary.json").read_text())
        for model in MODELS:
            if not np.isclose(observed[(condition, model)], summary[model]["uno_c_36m"], atol=1e-10):
                raise ValueError(f"Computed Uno C disagrees with {condition}/{model}")
    rng = np.random.default_rng(SEED + 1001)
    draws = {key: [] for key in observed}
    for iteration in range(BOOTSTRAP):
        values = {key: [] for key in observed}
        counts = np.bincount(rng.integers(0, len(data), len(data)), minlength=len(data))
        for indices, time, event, probability, risks in units:
            selected = np.repeat(np.arange(len(time)), counts[indices])
            selected_time = time[selected]
            selected_event = event[selected]
            selected_probability = probability[selected]
            for key, risk in risks.items():
                values[key].append(score(selected_time, selected_event, risk[selected],
                                         selected_probability))
        for key, fold_scores in values.items():
            draws[key].append(float(np.nanmean(fold_scores)))
        if (iteration + 1) % 25 == 0:
            print(f"Paired bootstrap {iteration + 1}/{BOOTSTRAP}", flush=True)
    result = {
        "method": "500 paired patient-cluster bootstrap resamples across all five repetitions, shared across all models and conditions; conditional on fitted models and fold assignments",
        "seed": SEED + 1001,
        "condition_uno_c": {
            condition: {
                model: {
                    "estimate": observed[(condition, model)],
                    "ci95": np.quantile(draws[(condition, model)], [0.025, 0.975]).tolist(),
                } for model in MODELS
            } for condition in conditions
        },
        "contrasts": {
            contrast: {
                model: {
                    "estimate": observed[(left, model)] - observed[(right, model)],
                    "ci95": np.quantile(np.asarray(draws[(left, model)]) -
                                        np.asarray(draws[(right, model)]), [0.025, 0.975]).tolist(),
                } for model in MODELS
            } for contrast, (left, right) in CONTRASTS.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
