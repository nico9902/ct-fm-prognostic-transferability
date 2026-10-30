#!/usr/bin/env python3
"""Matched E4 Cox probes on frozen global embeddings from CT input interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from extract_lung1_e1_embeddings import DIMENSIONS
from run_lung1_e4_regional_cox import aggregate, fit_one
from run_lung1_e3_frozen_linear import write_json
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, ROOT, cohort, locked_folds, sha256


MODELS = ("colipri", "tangerine", "radjepa", "ctclip")
CONDITIONS = ("original", "gtv_filled", "sham_filled")
EMBEDDINGS = ROOT / "outputs/lung1_e4_masked_volume/embeddings"
EMBEDDINGS_INPAINT = ROOT / "outputs/lung1_e4_masked_volume/embeddings_inpaint"
EMBEDDINGS_BLACK = ROOT / "outputs/lung1_e4_masked_volume/embeddings_black"
OUTPUT = ROOT / "outputs/lung1_e4_masked_volume/cox_full416"


def load_matrix(model: str, condition: str, data: pd.DataFrame) -> np.ndarray:
    source = (EMBEDDINGS_INPAINT if condition.endswith("inpaint") else
              EMBEDDINGS_BLACK if condition.endswith("black") else EMBEDDINGS)
    folder = source / model / condition
    vectors = []
    for patient_id in data.patient_id:
        vector = np.load(folder / f"{patient_id}.npy", allow_pickle=False)
        if vector.shape != (DIMENSIONS[model],) or not np.isfinite(vector).all():
            raise ValueError(f"{model}/{condition}/{patient_id}: invalid global embedding")
        vectors.append(vector.astype(np.float32))
    return np.stack(vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--fill-method", choices=("median", "inpaint", "black"), default="median")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-sham", action="store_true",
                        help="Refit all three conditions on patients with a valid same-shape control")
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()
    full_data = cohort(COHORT)
    folds = locked_folds(full_data, FOLDS)
    condition_set = {"median": CONDITIONS,
                     "inpaint": ("original", "gtv_inpaint", "sham_inpaint"),
                     "black": ("original", "gtv_black", "sham_black")}[args.fill_method]
    conditions = condition_set if args.include_sham else condition_set[:2]
    if args.include_sham:
        control = pd.read_csv(ROOT / "outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv",
                              dtype={"patient_id": str})
        usable = set(control.loc[control.sham_dx_voxels.notna(), "patient_id"])
        data = full_data.loc[full_data.patient_id.isin(usable)].reset_index(drop=True)
    else:
        data = full_data
    available_models = ("colipri",) if args.fill_method == "inpaint" else MODELS
    if args.only_model and args.only_model not in available_models:
        parser.error("only COLIPRI has the post-hoc inpaint sensitivity embeddings")
    models = (args.only_model,) if args.only_model else available_models
    for model in models:
        extraction = {"median": EMBEDDINGS, "inpaint": EMBEDDINGS_INPAINT,
                      "black": EMBEDDINGS_BLACK}[args.fill_method]
        if not (extraction / model / "COMPLETE").exists():
            raise ValueError(f"{model}: complete masked embedding extraction first")
    default_dir = {
        "median": "cox_control403" if args.include_sham else "cox_full416",
        "inpaint": "cox_inpaint_control403" if args.include_sham else "cox_inpaint_full416",
        "black": "cox_black_control403" if args.include_sham else "cox_black_full416",
    }[args.fill_method]
    output = args.output_dir or ROOT / "outputs/lung1_e4_masked_volume" / default_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS),
                "conditions": list(conditions), "models": list(available_models),
                "control_manifest_sha256": sha256(
                    ROOT / "outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv"),
                "input_manifests_sha256": {model: sha256(
                    ROOT / f"outputs/lung1_e2_inputs/{model}/original_ct/input_manifest.csv")
                    for model in MODELS},
                "outer_repetition": 0, "outer_folds": 5,
                "pca_components": 16, "penalties": [0.01, 0.1, 1.0],
                "penalty_selection": "3fold_event_stratified_uno_c_36m_within_outer_development",
                "primary_metric": "mean_outer_fold_uno_c_36m",
                "design": "separate_train_and_test_per_input_condition",
                "included_patient_ids_sha256": hashlib.sha256(
                    "\n".join(data.patient_id).encode()).hexdigest(),
                "n_patients": len(data), "control_complete_case": args.include_sham}
    if args.fill_method != "median":
        protocol["fill_method"] = args.fill_method
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Masked-volume Cox protocol differs; use a new output directory")
    protocol_tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    protocol_tmp.write_text(json.dumps(protocol, indent=2) + "\n")
    protocol_tmp.replace(path)
    keys = [(model, condition, fold) for model in available_models
            for condition in conditions for fold in range(5)]
    last_death_day = float(data.loc[data.os_event == 1, "os_time_days"].max())
    for model in models:
        for condition in conditions:
            missing = [fold for fold in range(5) if not (
                output / "fits" / model / condition / f"fold_{fold}.json").exists()]
            if not missing:
                continue
            matrix = load_matrix(model, condition, data)
            for fold in missing:
                result = fit_one(data, matrix, model, condition, fold, folds, last_death_day)
                write_json(output / "fits" / model / condition / f"fold_{fold}.json", result)
                print(f"{model} {condition} fold={fold} Uno36={result['metrics']['uno_c_36m']:.3f}",
                      flush=True)
    if not args.no_aggregate:
        aggregate(output, keys)
    print(f"Completed {sum((output / 'fits' / m / c / f'fold_{f}.json').exists() for m,c,f in keys)}/{len(keys)} fits")


if __name__ == "__main__":
    main()
