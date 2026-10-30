#!/usr/bin/env python3
"""E4: matched frozen-token regional linear Cox probes on locked LUNG1 folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from run_lung1_e3_frozen_linear import inner_assignments, write_json
from run_lung1_frozen_cox_e1 import (
    ALPHAS, COHORT, FOLDS, GRID, ROOT, cohort, fit_cox, fold_metrics,
    locked_folds, predict, select_penalty, sha256, uno_c,
)


MODELS = ("colipri", "tangerine", "radjepa", "ctclip")
ALL_MODELS = MODELS + ("dinov3",)
REGIONS = ("all", "tumor", "non_gtv", "peri20", "tumor_peri20")
GEOMETRY = ROOT / "outputs/lung1_e4_geometry"
TOKENS = ROOT / "outputs/lung1_e2_tokens"
OUTPUT = ROOT / "outputs/lung1_e4_regional_cox"
COMPONENTS = 16


def pooled_vectors(model: str, data: pd.DataFrame,
                   geometry: Path = GEOMETRY) -> dict[str, np.ndarray]:
    token_folder = TOKENS / model / "original_ct"
    manifest = pd.read_csv(token_folder / "token_manifest.csv",
                           dtype={"patient_id": str, "ct_series_uid": str})
    if manifest.patient_id.duplicated().any() or set(manifest.patient_id) != set(data.patient_id):
        raise ValueError(f"{model}: cached token cohort mismatch")
    manifest = manifest.set_index("patient_id").loc[data.patient_id]
    if not np.array_equal(manifest.ct_series_uid.to_numpy(), data.ct_series_uid.to_numpy()):
        raise ValueError(f"{model}: cached token series mismatch")
    result = {region: [] for region in REGIONS}
    for row in manifest.itertuples():
        patient_id = row.Index
        tokens = np.load(token_folder / row.token_path, allow_pickle=False).astype(np.float32)
        tumor = np.load(geometry / "weights" / model / f"{patient_id}.npy",
                        allow_pickle=False).astype(np.float32)
        peri = np.load(geometry / "peri20/weights" / model / f"{patient_id}.npy",
                       allow_pickle=False).astype(np.float32)
        if tokens.ndim != 2 or len(tumor) != len(tokens) or len(peri) != len(tokens):
            raise ValueError(f"{model}/{patient_id}: token/weight shape mismatch")
        if not (np.isfinite(tokens).all() and np.isfinite(tumor).all()
                and np.isfinite(peri).all()):
            raise ValueError(f"{model}/{patient_id}: nonfinite tokens or weights")
        weights = {"all": np.ones(len(tokens), dtype=np.float32),
                   "tumor": tumor,
                   "non_gtv": (tumor == 0).astype(np.float32),
                   "peri20": peri,
                   "tumor_peri20": np.minimum(tumor + peri, 1)}
        for region, weight in weights.items():
            total = float(weight.sum())
            if total <= 0:
                raise ValueError(f"{model}/{patient_id}: empty {region} pool")
            vector = np.einsum("n,nd->d", weight, tokens, optimize=True) / total
            result[region].append(vector)
    return {region: np.stack(vectors).astype(np.float32)
            for region, vectors in result.items()}


def fit_one(data: pd.DataFrame, matrix: np.ndarray, model: str, region: str,
            fold: int, folds: pd.DataFrame, last_death_day: float) -> dict:
    test_ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
    test = data.patient_id.isin(test_ids).to_numpy()
    development = ~test
    train_data = data.loc[development].reset_index(drop=True)
    eval_data = data.loc[test].reset_index(drop=True)
    inner = inner_assignments(train_data, fold, 0, 100)
    alpha, inner_scores = select_penalty(matrix[development], train_data, inner, COMPONENTS)
    train_time = train_data.os_time_days.to_numpy(dtype=float)
    train_event = train_data.os_event.to_numpy(dtype=int)
    test_time = eval_data.os_time_days.to_numpy(dtype=float)
    test_event = eval_data.os_event.to_numpy(dtype=int)
    fitted = fit_cox(matrix[development], train_time, train_event, alpha, COMPONENTS)
    risk, survival = predict(fitted, matrix[test], GRID)
    metrics = fold_metrics(train_time, train_event, test_time, test_event, risk, survival)
    metrics["harrell_c_full"] = float(concordance_index(test_time, -risk, test_event))
    metrics["uno_c_full"] = uno_c(train_time, train_event, test_time, test_event,
                                  risk, tau=last_death_day)
    predictions = [{"patient_id": str(pid), "os_time_days": float(time),
                    "os_event": int(event), "risk": float(score)}
                   for pid, time, event, score in zip(eval_data.patient_id, test_time,
                                                       test_event, risk)]
    return {"model": model, "region": region, "fold": fold,
            "n_train": len(train_data), "n_test": len(eval_data),
            "events_train_full": int(train_event.sum()), "alpha": float(alpha),
            "inner_uno_c_36m": float(max(inner_scores)),
            "metrics": {key: float(value) if np.isfinite(value) else None
                        for key, value in metrics.items()}, "predictions": predictions}


def aggregate(output: Path, keys: list[tuple[str, str, int]]) -> None:
    rows, predictions = [], []
    for model, region, fold in keys:
        path = output / "fits" / model / region / f"fold_{fold}.json"
        if not path.exists():
            continue
        fit = json.loads(path.read_text())
        rows.append({key: value for key, value in fit.items()
                     if key not in ("metrics", "predictions")} | fit["metrics"])
        predictions.extend({"model": model, "region": region, "fold": fold, **row}
                           for row in fit["predictions"])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "out_of_fold_predictions.csv", index=False)
    if len(rows) == len(keys):
        metric_names = ("uno_c_36m", "harrell_c_full", "uno_c_full", "auc_1y",
                        "auc_2y", "auc_3y", "ibs_3m_36m")
        frame.groupby(["model", "region"], as_index=False)[list(metric_names)].mean().to_csv(
            output / "summary.csv", index=False)
        (output / "COMPLETE").write_text(f"{len(keys)} fits complete\n")


def main() -> None:
    global COMPONENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--components", type=int, default=COMPONENTS)
    parser.add_argument("--exclude-patients", default="",
                        help="Comma-separated IDs removed before all fitting and evaluation")
    parser.add_argument("--only-model", choices=ALL_MODELS)
    parser.add_argument("--models", default=",".join(MODELS),
                        help="Comma-separated model set locked in this output directory")
    parser.add_argument("--geometry-dir", type=Path, default=GEOMETRY)
    parser.add_argument("--max-new-fits", type=int, default=0)
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()
    if args.components < 1:
        parser.error("--components must be positive")
    COMPONENTS = args.components
    models = tuple(args.models.split(","))
    if not set(models) <= set(ALL_MODELS) or (args.only_model and args.only_model not in models):
        parser.error("unknown model in --models or --only-model outside --models")
    geometry = args.geometry_dir
    if not (geometry / "COMPLETE").exists() or not (geometry / "peri20/COMPLETE").exists():
        raise ValueError("Complete GTV and peritumor mapping before E4 Cox fitting")
    full_data = cohort(COHORT)
    folds = locked_folds(full_data, FOLDS)
    exclusions = tuple(sorted(set(filter(None, args.exclude_patients.split(",")))))
    if not set(exclusions).issubset(set(full_data.patient_id)):
        raise ValueError("Unknown excluded patient ID")
    included = ~full_data.patient_id.isin(exclusions).to_numpy()
    data = full_data.loc[included].reset_index(drop=True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "gtv_mapping_sha256": sha256(geometry / "protocol.json"),
                "peri20_mapping_sha256": sha256(geometry / "peri20/protocol.json"),
                "gtv_patient_table_sha256": sha256(geometry / "patient_model_geometry.csv"),
                "peri20_patient_table_sha256": sha256(geometry / "peri20/patient_model_geometry.csv"),
                "token_manifests_sha256": {model: sha256(TOKENS / model / "original_ct/token_manifest.csv")
                                           for model in models},
                "models": list(models), "regions": list(REGIONS),
                "outer_repetition": 0, "outer_folds": 5,
                "pca_components": COMPONENTS, "penalties": list(ALPHAS),
                "penalty_selection": "3fold_event_stratified_uno_c_36m_within_outer_development",
                "primary_metric": "mean_outer_fold_uno_c_36m",
                "outside_region": "token_blocks_with_zero_gtv_footprint",
                "peritumor_region": "physical_0_to_20_mm_ring",
                "tumor_peritumor_region": "physical_union_gtv_plus_0_to_20_mm_ring"}
    if exclusions:
        protocol["excluded_patient_ids"] = list(exclusions)
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("E4 protocol differs; use a new output directory")
    write_json(path, protocol)
    keys = [(model, region, fold) for model in models
            for region in REGIONS for fold in range(5)]
    last_death_day = float(data.loc[data.os_event == 1, "os_time_days"].max())
    new_fits = 0
    for model in models:
        if args.only_model and model != args.only_model:
            continue
        missing = [(region, fold) for region in REGIONS for fold in range(5)
                   if not (output / "fits" / model / region / f"fold_{fold}.json").exists()]
        if not missing:
            continue
        matrices = {region: matrix[included] for region, matrix in
                    pooled_vectors(model, full_data, geometry).items()}
        for region, fold in missing:
            fit = fit_one(data, matrices[region], model, region, fold, folds, last_death_day)
            write_json(output / "fits" / model / region / f"fold_{fold}.json", fit)
            new_fits += 1
            print(f"{model} {region} fold={fold} Uno36={fit['metrics']['uno_c_36m']:.3f}",
                  flush=True)
            if args.max_new_fits and new_fits >= args.max_new_fits:
                if not args.no_aggregate:
                    aggregate(output, keys)
                return
    if not args.no_aggregate:
        aggregate(output, keys)
    print(f"Completed {sum((output / 'fits' / m / r / f'fold_{f}.json').exists() for m,r,f in keys)}/{len(keys)} fits")


if __name__ == "__main__":
    main()
