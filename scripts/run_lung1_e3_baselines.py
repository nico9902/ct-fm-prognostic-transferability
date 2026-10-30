#!/usr/bin/env python3
"""Matched E3 learning curves for stage/volume and GTV radiomics Cox baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from run_lung1_e3_frozen_linear import BUDGETS, N_DRAWS, inner_assignments, make_subsets, write_json
from run_lung1_frozen_cox_e1 import (
    ALPHAS, COHORT, FOLDS, GRID, ROOT, cohort, fit_cox, fold_metrics,
    locked_folds, predict, sha256, uno_c,
)


OUTPUT = ROOT / "outputs/lung1_e3_baselines"
BASELINES = ("stage_volume", "radiomics")


def features(data: pd.DataFrame, name: str, radiomics_path: Path) -> tuple[np.ndarray, dict]:
    if name == "stage_volume":
        stage = data.overall_stage.astype(str).str.lower()
        known = ("i", "ii", "iiia", "iiib")
        if not stage.isin((*known, "nan")).all():
            raise ValueError("Unexpected clinical stage label")
        volume = np.log1p(pd.to_numeric(data.gtv_volume_cm3, errors="raise").to_numpy(dtype=float))
        array = np.column_stack([(stage == label).to_numpy(dtype=float)
                                 for label in ("ii", "iiia", "iiib", "nan")] + [volume])
        return array, {"columns": ["stage_II", "stage_IIIa", "stage_IIIb",
                                   "stage_missing", "log1p_gtv_volume_cm3"],
                       "stage_encoding": "one-hot with stage I reference; missingness indicator"}
    table = pd.read_csv(radiomics_path, dtype={"patient_id": str}).set_index("patient_id")
    if table.index.duplicated().any() or set(table.index) != set(data.patient_id):
        raise ValueError("Radiomics table does not match E3 cohort IDs")
    columns = sorted(column for column in table if column.startswith("original_"))
    if not columns:
        raise ValueError("No PyRadiomics features found")
    array = table.loc[data.patient_id, columns].to_numpy(dtype=float)
    if np.isinf(array).any():
        array[np.isinf(array)] = np.nan
    return array, {"features_sha256": sha256(radiomics_path), "n_features": len(columns),
                   "columns": columns, "image_type": "PyRadiomics Original"}


def impute(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    other = np.asarray(other, dtype=float)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return np.where(np.isfinite(train), train, medians), np.where(np.isfinite(other), other, medians)


def select_alpha(x: np.ndarray, subset: pd.DataFrame, inner: pd.DataFrame) -> tuple[float, float]:
    scores = []
    for alpha in ALPHAS:
        fold_scores = []
        for inner_fold in range(3):
            validation = subset.patient_id.isin(
                inner.loc[inner.inner_fold == inner_fold, "patient_id"]).to_numpy()
            training = ~validation
            train_x, validation_x = impute(x[training], x[validation])
            train = subset.iloc[np.flatnonzero(training)]
            valid = subset.iloc[np.flatnonzero(validation)]
            fitted = fit_cox(train_x, train.os_time_days.to_numpy(float),
                             train.os_event.to_numpy(int), alpha, 0)
            risk, _ = predict(fitted, validation_x, GRID)
            score = uno_c(train.os_time_days.to_numpy(float), train.os_event.to_numpy(int),
                          valid.os_time_days.to_numpy(float), valid.os_event.to_numpy(int), risk)
            if not np.isfinite(score):
                raise ValueError("Undefined inner Uno C")
            fold_scores.append(score)
        scores.append(float(np.mean(fold_scores)))
    best = int(np.argmax(scores))
    return ALPHAS[best], scores[best]


def one_fit(data: pd.DataFrame, matrix: np.ndarray, train_indices: np.ndarray,
            test_indices: np.ndarray, development_indices: np.ndarray,
            name: str, fold: int, draw: int, budget: int, last_death: float) -> dict:
    subset = data.iloc[train_indices].reset_index(drop=True)
    inner = inner_assignments(subset, fold, draw, budget)
    alpha, inner_score = select_alpha(matrix[train_indices], subset, inner)
    train_x, test_x = impute(matrix[train_indices], matrix[test_indices])
    fitted = fit_cox(train_x, subset.os_time_days.to_numpy(float),
                     subset.os_event.to_numpy(int), alpha, 0)
    risk, survival = predict(fitted, test_x, GRID)
    test = data.iloc[test_indices]
    development = data.iloc[development_indices]
    eval_time = development.os_time_days.to_numpy(float)
    eval_event = development.os_event.to_numpy(int)
    test_time = test.os_time_days.to_numpy(float)
    test_event = test.os_event.to_numpy(int)
    metrics = fold_metrics(eval_time, eval_event, test_time, test_event, risk, survival)
    metrics["harrell_c_full"] = float(concordance_index(test_time, -risk, test_event))
    metrics["uno_c_full"] = uno_c(eval_time, eval_event, test_time, test_event, risk,
                                  tau=last_death)
    if not np.isfinite(metrics["uno_c_36m"]):
        raise ValueError(f"Undefined primary metric: {name}, {fold}, {draw}, {budget}")
    predictions = [
        {"patient_id": str(pid), "os_time_days": float(time), "os_event": int(event),
         "risk": float(score), "risk_36m": float(-np.log(np.clip(curve[-1], 1e-12, 1.0)))}
        for pid, time, event, score, curve in zip(test.patient_id, test_time, test_event, risk, survival)
    ]
    return {"model": name, "fold": fold, "draw": draw, "budget_pct": budget,
            "n_train": len(subset), "n_test": len(test),
            "events_train_full": int(subset.os_event.sum()),
            "events_train_36m": int(((subset.os_event == 1) &
                                     (subset.os_time_days <= GRID[-1])).sum()),
            "censored_train": int((subset.os_event == 0).sum()),
            "alpha": float(alpha), "inner_uno_c_36m": float(inner_score),
            "metrics": {key: float(value) if np.isfinite(value) else None
                        for key, value in metrics.items()}, "predictions": predictions}


def aggregate(output: Path, keys: list[tuple]) -> None:
    rows, predictions = [], []
    for name, fold, draw, budget in keys:
        path = output / "fits" / name / f"fold_{fold}" / f"draw_{draw}_budget_{budget}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows.append({key: value for key, value in result.items() if key not in ("metrics", "predictions")}
                    | result["metrics"])
        predictions.extend({"model": name, "fold": fold, "draw": draw, "budget_pct": budget, **row}
                           for row in result["predictions"])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "out_of_fold_predictions.csv", index=False)
    write_json(output / "progress.json", {"completed_fits": len(rows), "expected_fits": len(keys)})
    if len(rows) != len(keys):
        (output / "COMPLETE").unlink(missing_ok=True)
        return
    metrics = ["uno_c_36m", "harrell_c_full", "uno_c_full", "auc_1y", "auc_2y",
               "auc_3y", "ibs_3m_36m", "n_train", "events_train_full",
               "events_train_36m", "censored_train"]
    draws = frame.groupby(["model", "budget_pct", "draw"], as_index=False)[metrics].mean()
    draws.to_csv(output / "draw_metrics.csv", index=False)
    draws.groupby(["model", "budget_pct"], as_index=False)[metrics].mean().to_csv(
        output / "summary.csv", index=False)
    (output / "COMPLETE").write_text(f"{len(rows)} fits complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--radiomics-features", type=Path, default=OUTPUT / "radiomics_features.csv")
    parser.add_argument("--only-model", choices=BASELINES)
    parser.add_argument("--max-new-fits", type=int, default=0)
    args = parser.parse_args()
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    source = ROOT / "outputs/lung1_e3_frozen_linear/subsets.csv"
    subsets, table = make_subsets(data, folds)
    if not pd.read_csv(source, dtype={"patient_id": str}).equals(table):
        raise ValueError("E3 locked subsets have changed")
    names = (args.only_model,) if args.only_model else BASELINES
    matrices, provenance = {}, {}
    for name in names:
        matrices[name], provenance[name] = features(data, name, args.radiomics_features)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "subsets_sha256": sha256(source), "outer_repetition": 0,
                "budgets_pct": list(BUDGETS), "subset_draws": N_DRAWS,
                "penalties": list(ALPHAS), "penalty_selection": "3fold_event_stratified_uno_c_36m_within_budget",
                "cox": "standardized original features, L2 ridge Cox, no PCA",
                "missing_features": "median imputed using fitting patients only, at each inner and final fit",
                "primary_metric": "mean_test_fold_uno_c_36m",
                "censoring_for_outer_evaluation": "fixed_outer_development_pool;never_used_to_fit_or_select_cox",
                "baselines": provenance}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text())
        for key, value in protocol.items():
            if key == "baselines":
                for name, detail in value.items():
                    if name in saved["baselines"] and saved["baselines"][name] != detail:
                        raise ValueError("Feature provenance changed; use a new output directory")
                    saved["baselines"][name] = detail
            elif saved[key] != value:
                raise ValueError("E3 baseline protocol changed; use a new output directory")
        write_json(protocol_path, saved)
    else:
        write_json(protocol_path, protocol)
    last_death = float(data.loc[data.os_event == 1, "os_time_days"].max())
    keys = [(name, fold, draw, budget) for name in BASELINES for fold in range(5)
            for budget in BUDGETS for draw in range(1 if budget == 100 else N_DRAWS)]
    new_fits = 0
    for name in names:
        for fold in range(5):
            test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
            test_indices = np.flatnonzero(data.patient_id.isin(test_ids).to_numpy())
            development_indices = np.flatnonzero(~data.patient_id.isin(test_ids).to_numpy())
            for budget in BUDGETS:
                for draw in range(1 if budget == 100 else N_DRAWS):
                    path = output / "fits" / name / f"fold_{fold}" / f"draw_{draw}_budget_{budget}.json"
                    if path.exists():
                        continue
                    result = one_fit(data, matrices[name], subsets[(fold, draw, budget)],
                                     test_indices, development_indices, name, fold, draw,
                                     budget, last_death)
                    write_json(path, result)
                    new_fits += 1
                    if args.max_new_fits and new_fits >= args.max_new_fits:
                        aggregate(output, keys)
                        return
            print(f"{name}: fold {fold + 1}/5 complete", flush=True)
    aggregate(output, keys)


if __name__ == "__main__":
    main()
