#!/usr/bin/env python3
"""Paired patient-cluster bootstrap for completed E2 adaptation rungs."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_lung1_e1_conditions import score  # noqa: E402
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, censor_survival, cohort, locked_folds  # noqa: E402

MODELS = ("tangerine", "radjepa", "colipri", "ctclip")
LINEAR = ROOT / "outputs/lung1_e1_frozen_cox_four_models"
METHODS = {"mlp": ROOT / "outputs/lung1_e2_mlp",
           "sqa": ROOT / "outputs/lung1_e2_sqa",
           "lora": ROOT / "outputs/lung1_e2_encoder_adaptation/lora",
           "partial": ROOT / "outputs/lung1_e2_encoder_adaptation/partial"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lung1_e2_comparison/paired_uno_c.json")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    linear = pd.read_csv(LINEAR / "out_of_fold_predictions.csv")
    predictions = {("linear", model): linear.loc[linear.model.eq(model)].copy() for model in MODELS}
    for method, root in METHODS.items():
        if all((root / model / "summary.json").is_file() and
               json.loads((root / model / "summary.json").read_text()).get("completed_folds") == 25
               for model in MODELS):
            for model in MODELS:
                predictions[(method, model)] = pd.read_csv(root / model / "out_of_fold_predictions.csv")
    if len(predictions) == len(MODELS):
        raise ValueError("No complete adaptation rung is available")
    groups = {key: {(int(rep), int(fold)): part.sort_values("patient_id").reset_index(drop=True)
                    for (rep, fold), part in frame.groupby(["repetition", "fold"])}
              for key, frame in predictions.items()}
    keys = list(predictions)
    index = {patient_id: i for i, patient_id in enumerate(data.patient_id)}
    units = []
    for repetition in range(5):
        for fold in range(5):
            reference = groups[keys[0]][(repetition, fold)]
            test_ids = set(folds.loc[(folds.repetition.eq(repetition)) &
                                     (folds.fold.eq(fold)), "patient_id"])
            if set(reference.patient_id) != test_ids:
                raise ValueError("Reference predictions differ from locked test fold")
            training = data.loc[~data.patient_id.isin(test_ids)]
            time = reference.os_time_days.to_numpy()
            event = reference.os_event.to_numpy()
            g = censor_survival(training.os_time_days.to_numpy(), training.os_event.to_numpy())
            risk = {}
            for key in keys:
                part = groups[key][(repetition, fold)]
                if not (part.patient_id.equals(reference.patient_id)
                        and np.array_equal(part.os_time_days.to_numpy(), time)
                        and np.array_equal(part.os_event.to_numpy(), event)):
                    raise ValueError(f"Misaligned predictions: {key}/{repetition}/{fold}")
                risk[key] = part.risk.to_numpy()
            units.append((np.array([index[patient_id] for patient_id in reference.patient_id]),
                          time, event, g(time), risk))
    observed = {key: float(np.mean([score(t, e, risk[key], g) for _, t, e, g, risk in units]))
                for key in keys}
    rng = np.random.default_rng(20261006)
    draws = {key: [] for key in keys}
    for _ in range(args.bootstrap):
        counts = np.bincount(rng.integers(0, len(data), len(data)), minlength=len(data))
        values = {key: [] for key in keys}
        for indices, time, event, censor, risk in units:
            selected = np.repeat(np.arange(len(time)), counts[indices])
            for key in keys:
                values[key].append(score(time[selected], event[selected], risk[key][selected],
                                         censor[selected]))
        for key in keys:
            draws[key].append(float(np.nanmean(values[key])))
    result = {"method": "Paired patient-cluster bootstrap, shared over all five repetitions; conditional on trained folds",
              "bootstrap": args.bootstrap, "seed": 20261006, "conditions": {}, "contrasts": {}}
    for (method, model), value in observed.items():
        result["conditions"].setdefault(method, {})[model] = {
            "uno_c_36m": value, "ci95": np.quantile(draws[(method, model)], [0.025, 0.975]).tolist()}
    for method in METHODS:
        if (method, MODELS[0]) not in observed:
            continue
        for model in MODELS:
            diff = np.asarray(draws[(method, model)]) - np.asarray(draws[("linear", model)])
            result["contrasts"].setdefault(f"{method}_minus_linear", {})[model] = {
                "uno_c_36m": observed[(method, model)] - observed[("linear", model)],
                "ci95": np.quantile(diff, [0.025, 0.975]).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "lung1-prognostic-transferability"),
                     name="E2-paired-comparison", group="lung1-e2-original-ct",
                     job_type="paired-comparison", mode=os.environ.get("WANDB_MODE", "offline"),
                     dir=str(args.output.parent), config={"bootstrap": args.bootstrap, "methods": list(result["conditions"])})
    for method, models in result["conditions"].items():
        for model, estimates in models.items():
            wandb.summary[f"{method}/{model}/uno_c_36m"] = estimates["uno_c_36m"]
    for contrast, models in result["contrasts"].items():
        for model, estimates in models.items():
            wandb.summary[f"{contrast}/{model}/uno_c_36m"] = estimates["uno_c_36m"]
    run.finish()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
