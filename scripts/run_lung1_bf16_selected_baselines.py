#!/usr/bin/env python3
"""Literature baselines on the locked one-repetition 5-fold LUNG1 E1/E2 split.

Stage + tumor volume and GTV radiomics (the E3 baselines) use the same patients,
train/validation/test split, penalty grid, validation Uno C selection, pooled
metrics and out-of-fold prediction format as `run_lung1_bf16_selected_heads.py`
with `--method linear`. Cox uses all standardized covariates (no PCA).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_lung1_e2_encoder_bf16_lightning import aggregate, write_json  # noqa: E402
from run_lung1_e3_baselines import BASELINES, features, impute  # noqa: E402
from run_lung1_frozen_cox_e1 import (  # noqa: E402
    ALPHAS, GRID, cohort, fit_cox, fold_metrics, inner_assignments,
    locked_folds, predict, sha256, uno_c,
)

COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
FOLDS = ROOT / "configs/lung1_e1_folds.csv"
RADIOMICS = ROOT / "outputs/lung1_e3_baselines/radiomics_features.csv"
SELECTED = ROOT / "outputs/lung1_bf16_selected"
OUTPUT = ROOT / "outputs/lung1_bf16_selected_baselines"
FM_METHODS = ("linear", "mlp", "sqa", "lora", "partial")
FM_MODELS = ("tangerine", "radjepa", "colipri", "ctclip", "dinov3")
BOOTSTRAP_SEED = 20261015


def run_baseline(name: str, data: pd.DataFrame, folds: pd.DataFrame,
                 inner: pd.DataFrame, output_root: Path, max_folds: int) -> None:
    matrix, provenance = features(data, name, RADIOMICS)
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": name, "method": "baseline_ridge_cox",
                "outer_repetition": 0, "outer_folds": 5, "validation_inner_fold": 0,
                "selection": "maximum_validation_uno_c_36m_earliest_tie",
                "cox": "train-only median imputation and standardization; L2 Cox on all "
                       "covariates, Efron ties, L-BFGS, Breslow baseline; no PCA",
                "penalties": list(ALPHAS), "refit_after_selection": False,
                "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "features": provenance}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a new output root")
    write_json(protocol_path, protocol)
    completed = {int(path.stem.split("_")[-1]) for path in (output / "folds").glob("fold_*.json")}
    t = data.os_time_days.to_numpy(dtype=float)
    e = data.os_event.to_numpy(dtype=int)
    for fold in range(max_folds):
        if fold in completed:
            print(f"{name} fold={fold} already complete", flush=True)
            continue
        test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        val_ids = set(inner.loc[(inner.repetition == 0) & (inner.fold == fold) &
                                (inner.inner_fold == 0), "patient_id"])
        test = data.patient_id.isin(test_ids).to_numpy()
        val = data.patient_id.isin(val_ids).to_numpy()
        train = ~(test | val)
        if (test & val).any() or train.sum() + val.sum() + test.sum() != len(data):
            raise ValueError("Invalid patient split")
        started = time.monotonic()
        x_train, x_val = impute(matrix[train], matrix[val])
        _, x_test = impute(matrix[train], matrix[test])
        scores, candidates = [], []
        for penalty in ALPHAS:
            fitted = fit_cox(x_train, t[train], e[train], penalty, 0)
            val_risk, _ = predict(fitted, x_val, GRID)
            scores.append(float(uno_c(t[train], e[train], t[val], e[val], val_risk)))
            candidates.append(fitted)
        if not np.isfinite(scores).all():
            raise ValueError("Undefined validation Uno C")
        best = int(np.argmax(scores))
        risk, survival = predict(candidates[best], x_test, GRID)
        metrics = fold_metrics(t[train], e[train], t[test], e[test], risk, survival)
        row = {"model": name, "method": "baseline", "repetition": 0, "fold": fold,
               "n_train": int(train.sum()), "n_validation": int(val.sum()),
               "n_test": int(test.sum()), "elapsed_seconds": time.monotonic() - started,
               "selected_penalty": ALPHAS[best], "selected_validation_uno_c_36m": scores[best],
               "validation_uno_c_by_penalty": dict(zip(map(str, ALPHAS), scores)), **metrics}
        predictions = []
        for i, index in enumerate(np.flatnonzero(test)):
            prediction = {"model": name, "method": "baseline", "repetition": 0, "fold": fold,
                          "patient_id": data.patient_id.iloc[index],
                          "os_time_days": float(t[index]), "os_event": int(e[index]),
                          "risk": float(risk[i]),
                          "risk_36m": float(-np.log(np.clip(survival[i, -1], 1e-12, 1.0)))}
            prediction.update({f"survival_{j:02d}": float(value)
                               for j, value in enumerate(survival[i])})
            predictions.append(prediction)
        write_json(output / "folds" / f"fold_{fold}.json",
                   {"metrics": row, "predictions": predictions})
        count = aggregate(output, data)
        print(f"{name} fold={fold} complete; penalty={ALPHAS[best]}; "
              f"test Uno C={metrics['uno_c_36m']:.3f}; completed={count}/5", flush=True)


def pooled_risk(path: Path, data: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
    if len(frame) != len(data) or set(frame.index) != set(data.patient_id):
        raise ValueError(f"{path} does not cover the 416 patients once")
    return frame.loc[data.patient_id, "risk_36m"].to_numpy(dtype=float)


def compare(data: pd.DataFrame, output_root: Path, n_boot: int) -> None:
    """Paired patient bootstrap of pooled Uno C; censoring curve fixed to the full cohort."""
    t = data.os_time_days.to_numpy(dtype=float)
    e = data.os_event.to_numpy(dtype=int)
    risks = {f"baseline/{name}": pooled_risk(output_root / name / "out_of_fold_predictions.csv", data)
             for name in BASELINES}
    for method in FM_METHODS:
        for model in FM_MODELS:
            path = SELECTED / method / model / "out_of_fold_predictions.csv"
            if path.exists():
                risks[f"{method}/{model}"] = pooled_risk(path, data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(data), len(data)) for _ in range(n_boot)]
    point = {key: float(uno_c(t, e, t, e, value)) for key, value in risks.items()}
    boot = {key: np.array([uno_c(t, e, t[d], e[d], value[d]) for d in draws])
            for key, value in risks.items()}

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    result = {"metric": "pooled_uno_c_36m", "risk": "negative_log_predicted_survival_at_36_months",
              "bootstrap": n_boot, "seed": BOOTSTRAP_SEED,
              "note": "conditional on fitted fold models; no multiplicity adjustment",
              "conditions": {key: {"uno_c_36m": point[key], "ci95": interval(boot[key])}
                             for key in risks},
              "differences": {}}
    for base in (key for key in risks if key.startswith("baseline/")):
        for other in risks:
            if other == base:
                continue
            delta = boot[other] - boot[base]
            result["differences"][f"{other} - {base}"] = {
                "difference": point[other] - point[base], "ci95": interval(delta)}
    write_json(output_root / "paired_uno_c.json", result)
    print(json.dumps(result["conditions"], indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-model", choices=BASELINES)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5:
        parser.error("max-folds must be 1 to 5")
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    inner = inner_assignments(data, folds)
    for name in (args.only_model,) if args.only_model else BASELINES:
        run_baseline(name, data, folds, inner, args.output_root, args.max_folds)
    if all((args.output_root / name / "summary.json").exists() for name in BASELINES):
        compare(data, args.output_root, args.bootstrap)
        (args.output_root / "COMPLETE").write_text("2 baselines x 5 folds complete\n")


if __name__ == "__main__":
    main()
