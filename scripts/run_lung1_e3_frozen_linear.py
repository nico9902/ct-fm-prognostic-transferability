#!/usr/bin/env python3
"""E3: nested label-budget curves for frozen LUNG1 embeddings and linear Cox.

Uses repetition zero of the locked E1 outer folds. Every budget includes all
labels used for Cox fitting and penalty selection; the outer development cohort
supplies a fixed censoring estimate solely for held-out metric calculation.
Each completed fit is saved separately so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split

from run_lung1_frozen_cox_e1 import (
    ALPHAS, COHORT, FOLDS, GRID, HORIZONS, ROOT, cohort, fit_cox,
    fold_metrics, load_embeddings, locked_folds, predict, select_penalty,
    sha256, strata, uno_c,
)


MODELS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3", "pai", "hosny")
BUDGETS = (10, 25, 50, 100)
N_DRAWS = 10
PCA_COMPONENTS = 8
SEED = 20261009
OUTPUT = ROOT / "outputs/lung1_e3_frozen_linear"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_size(fraction: int, population: int) -> int:
    return int(np.floor(population * fraction / 100 + 0.5))


def make_subsets(data: pd.DataFrame, folds: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    subsets = {}
    records = []
    for fold in range(5):
        test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        available = np.flatnonzero(~data.patient_id.isin(test_ids).to_numpy())
        labels = strata(data.iloc[available])
        events = data.os_event.to_numpy()[available]
        for draw in range(N_DRAWS):
            seed = SEED + 100 * fold + draw
            half, _ = train_test_split(
                np.arange(len(available)), train_size=sample_size(50, len(available)),
                stratify=labels, random_state=seed,
            )
            quarter, _ = train_test_split(
                half, train_size=sample_size(25, len(available)),
                stratify=labels[half], random_state=seed + 1000,
            )
            # The smallest early-stage censored stratum has only 1–2 members at
            # 25%. Stratify the final draw by event status to keep CV feasible.
            tenth, _ = train_test_split(
                quarter, train_size=sample_size(10, len(available)),
                stratify=events[quarter], random_state=seed + 2000,
            )
            chosen = {10: tenth, 25: quarter, 50: half}
            if draw == 0:
                chosen[100] = np.arange(len(available))
            for budget, local in chosen.items():
                indices = np.sort(available[local])
                subsets[(fold, draw, budget)] = indices
                for index in indices:
                    records.append({"fold": fold, "draw": draw, "budget_pct": budget,
                                    "patient_id": data.patient_id.iloc[index]})
            if not (set(tenth) <= set(quarter) <= set(half)):
                raise AssertionError("Budget subsets are not nested")
    return subsets, pd.DataFrame(records).sort_values(
        ["fold", "draw", "budget_pct", "patient_id"]).reset_index(drop=True)


def inner_assignments(subset: pd.DataFrame, fold: int, draw: int, budget: int) -> pd.DataFrame:
    events = subset.os_event.to_numpy()
    if min(np.bincount(events, minlength=2)) < 3:
        raise ValueError(f"Fold {fold}, draw {draw}, budget {budget}: fewer than 3 in an event class")
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True,
        random_state=SEED + 10000 + 1000 * fold + 100 * draw + budget,
    )
    records = []
    for inner_fold, (_, validation) in enumerate(splitter.split(subset, events)):
        for index in validation:
            records.append({"inner_fold": inner_fold, "patient_id": subset.patient_id.iloc[index]})
    return pd.DataFrame(records)


def fit_one(
    data: pd.DataFrame, matrix: np.ndarray, subset_indices: np.ndarray,
    test_indices: np.ndarray, development_indices: np.ndarray,
    model: str, fold: int, draw: int, budget: int, last_death_day: float,
) -> dict:
    subset = data.iloc[subset_indices].reset_index(drop=True)
    inner = inner_assignments(subset, fold, draw, budget)
    alpha, inner_scores = select_penalty(matrix[subset_indices], subset, inner, PCA_COMPONENTS)
    train_time = subset.os_time_days.to_numpy(dtype=float)
    train_event = subset.os_event.to_numpy(dtype=int)
    test = data.iloc[test_indices]
    test_time = test.os_time_days.to_numpy(dtype=float)
    test_event = test.os_event.to_numpy(dtype=int)
    development = data.iloc[development_indices]
    eval_time = development.os_time_days.to_numpy(dtype=float)
    eval_event = development.os_event.to_numpy(dtype=int)
    fitted = fit_cox(matrix[subset_indices], train_time, train_event, alpha, PCA_COMPONENTS)
    risk, survival = predict(fitted, matrix[test_indices], GRID)
    metrics = fold_metrics(eval_time, eval_event, test_time, test_event, risk, survival)
    metrics["harrell_c_full"] = float(concordance_index(test_time, -risk, test_event))
    metrics["uno_c_full"] = uno_c(
        eval_time, eval_event, test_time, test_event, risk, tau=last_death_day,
    )
    if not np.isfinite(metrics["uno_c_36m"]) or not np.isfinite(metrics["harrell_c_full"]):
        raise ValueError(f"Undefined primary metric: {model}, fold {fold}, draw {draw}, {budget}%")
    predictions = [
        {"patient_id": str(patient_id), "os_time_days": float(time),
         "os_event": int(event), "risk": float(score),
         "risk_36m": float(-np.log(np.clip(curve[-1], 1e-12, 1.0)))}
        for patient_id, time, event, score, curve in zip(
            test.patient_id, test_time, test_event, risk, survival,
        )
    ]
    return {
        "model": model, "fold": fold, "draw": draw, "budget_pct": budget,
        "n_train": len(subset), "n_test": len(test),
        "events_train_full": int(train_event.sum()),
        "events_train_36m": int(((train_event == 1) & (train_time <= HORIZONS[-1])).sum()),
        "censored_train": int((train_event == 0).sum()),
        "alpha": float(alpha), "inner_uno_c_36m": float(max(inner_scores)),
        "metrics": {key: float(value) if np.isfinite(value) else None
                    for key, value in metrics.items()},
        "predictions": predictions,
    }


def aggregate(output: Path, keys: list[tuple], expected: int) -> None:
    rows = []
    predictions = []
    for model, fold, draw, budget in keys:
        path = output / "fits" / model / f"fold_{fold}" / f"draw_{draw}_budget_{budget}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows.append({key: value for key, value in result.items() if key not in ("metrics", "predictions")}
                    | result["metrics"])
        predictions.extend(
            {"model": model, "fold": fold, "draw": draw, "budget_pct": budget, **row}
            for row in result["predictions"]
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "out_of_fold_predictions.csv", index=False)
    write_json(output / "progress.json", {"completed_fits": len(rows), "expected_fits": expected})
    if len(rows) != expected:
        return
    metric_columns = ("uno_c_36m", "harrell_c_full", "uno_c_full", "auc_1y", "auc_2y",
                      "auc_3y", "ibs_3m_36m")
    draws = metrics.groupby(["model", "budget_pct", "draw"], as_index=False)[
        list(metric_columns) + ["n_train", "events_train_full", "events_train_36m", "censored_train"]
    ].mean()
    draws.to_csv(output / "draw_metrics.csv", index=False)
    summary = draws.groupby(["model", "budget_pct"], as_index=False)[
        list(metric_columns) + ["n_train", "events_train_full", "events_train_36m", "censored_train"]
    ].mean()
    summary.to_csv(output / "summary.csv", index=False)
    (output / "COMPLETE").write_text(f"{len(rows)} fits complete\n")


def main() -> None:
    global PCA_COMPONENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--components", type=int, default=PCA_COMPONENTS,
                        help="Fixed PCA dimension for every model and budget")
    parser.add_argument("--only-model", choices=MODELS,
                        help="Run one independent model queue")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Leave aggregation to a later complete run")
    parser.add_argument("--max-new-fits", type=int, default=0,
                        help="Stop after this many new fits; zero runs all missing fits")
    args = parser.parse_args()
    if args.max_new_fits < 0:
        parser.error("--max-new-fits must be nonnegative")
    if args.components < 1:
        parser.error("--components must be positive")
    PCA_COMPONENTS = args.components
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    features = {}
    provenance = {}
    for model in MODELS:
        manifest = ROOT / "outputs/lung1_e1_embeddings" / model / "original_ct" / "embedding_manifest.csv"
        name, matrix, manifest_hash, matrix_hash = load_embeddings(f"{model}={manifest}", data)
        features[name] = matrix
        provenance[name] = {"manifest_sha256": manifest_hash, "matrix_sha256": matrix_hash,
                            "dimension": int(matrix.shape[1])}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
        "outer_repetition": 0, "outer_folds": 5, "budgets_pct": list(BUDGETS),
        "subset_draws": N_DRAWS, "subset_seed": SEED,
        "subset_stratification": "event_and_early_late_stage_at_50_and_25pct;event_at_10pct",
        "nested_subsets": True, "pca_components_fixed": PCA_COMPONENTS,
        "penalties": list(ALPHAS), "penalty_selection": "3fold_event_stratified_uno_c_36m_within_budget",
        "primary_metric": "mean_test_fold_uno_c_36m",
        "censoring_for_outer_evaluation": "fixed_outer_development_pool;never_used_to_fit_or_select_cox",
        "secondary_metrics": ["harrell_c_full", "uno_c_full", "auc_1y", "auc_2y", "auc_3y", "ibs_3m_36m"],
        "model_embeddings": provenance,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text())
        # A later encoder may be appended; everything already locked must match.
        saved_models = saved.get("model_embeddings", {})
        if ({k: v for k, v in saved.items() if k != "model_embeddings"}
                != {k: v for k, v in protocol.items() if k != "model_embeddings"}
                or any(protocol["model_embeddings"].get(m) != v for m, v in saved_models.items())):
            raise ValueError("E3 protocol differs from saved protocol; choose a new output directory")
        if set(protocol["model_embeddings"]) - set(saved_models):
            write_json(protocol_path, protocol)
    else:
        write_json(protocol_path, protocol)
    subsets, table = make_subsets(data, folds)
    subset_path = output / "subsets.csv"
    if subset_path.exists():
        if not pd.read_csv(subset_path, dtype={"patient_id": str}).equals(table):
            raise ValueError("Saved E3 subsets differ from deterministic subsets")
    else:
        table.to_csv(subset_path, index=False)
    last_death_day = float(data.loc[data.os_event == 1, "os_time_days"].max())
    keys = []
    for model in MODELS:
        for fold in range(5):
            for draw in range(N_DRAWS):
                for budget in BUDGETS:
                    if budget == 100 and draw != 0:
                        continue
                    keys.append((model, fold, draw, budget))
    new_fits = 0
    run_keys = [key for key in keys if args.only_model is None or key[0] == args.only_model]
    for model, fold, draw, budget in run_keys:
        path = output / "fits" / model / f"fold_{fold}" / f"draw_{draw}_budget_{budget}.json"
        if path.exists():
            continue
        test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test_indices = np.flatnonzero(data.patient_id.isin(test_ids).to_numpy())
        development_indices = np.flatnonzero(~data.patient_id.isin(test_ids).to_numpy())
        result = fit_one(data, features[model], subsets[(fold, draw, budget)],
                         test_indices, development_indices,
                         model, fold, draw, budget, last_death_day)
        write_json(path, result)
        new_fits += 1
        print(f"{model} fold={fold} draw={draw} budget={budget}% "
              f"n={result['n_train']} deaths={result['events_train_full']} "
              f"Uno36={result['metrics']['uno_c_36m']:.3f}", flush=True)
        if args.max_new_fits and new_fits >= args.max_new_fits:
            break
    if not args.no_aggregate:
        aggregate(output, keys, len(keys))
    print(f"Completed {sum((output / 'fits' / m / f'fold_{f}' / f'draw_{d}_budget_{b}.json').exists() for m,f,d,b in keys)}/{len(keys)} fits")


if __name__ == "__main__":
    main()
