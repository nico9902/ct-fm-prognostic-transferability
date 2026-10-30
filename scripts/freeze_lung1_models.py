#!/usr/bin/env python3
"""Phase 4.3: refit every external-validation model on all 416 LUNG1 patients and freeze it.

Pre-specified in docs/lung1_phase4_protocol.md. Each model is refitted with the selection
procedure it used internally, now on the whole cohort: penalties from the same grid by
3-fold event-stratified CV (one deterministic split of the 416 patients, shared by all
models), and imputation, scaling and PCA fitted on the 416.

* ladder models (C0-C4, C3-pai, C3-hosny_score): run_lung1_ceiling_ladder.py procedure
  (clinical penalty x shared radiomics/image penalty, PCA16 image/radiomics blocks);
* FM / Pai / Hosny alone: E3 procedure (standardization, PCA16, one penalty);
* stage + volume and radiomics baselines: E3 baseline procedure (median imputation,
  standardized features, one penalty, no PCA);
* Hosny zero-shot: nothing to fit.

Nothing from NSCLC-Radiogenomics is read. Every fitted model is pickled and hashed in
outputs/phase4_frozen_models/manifest.json.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_lung1_ceiling_ladder import (
    COMPONENTS, FMS, RADIOMICS, Block, clinical_block, fit as fit_ladder,
)
from run_lung1_e3_baselines import features as baseline_features, impute, select_alpha
from run_lung1_e3_frozen_linear import inner_assignments
from run_lung1_frozen_cox_e1 import (
    ALPHAS, COHORT, ROOT, cohort, fit_cox, load_embeddings, select_penalty, sha256, uno_c,
)

OUTPUT = ROOT / "outputs/phase4_frozen_models"
EMBEDDINGS = ROOT / "outputs/lung1_e1_embeddings"
FULL_FIT_FOLD = 5  # inner split id for the whole-cohort refit (outer folds used 0-4)
TUMOUR = ("pai", "hosny")


def ladder_models() -> dict[str, tuple[str, ...]]:
    models = {"C0": ("clinical0",), "C1": ("clinical",), "C2": ("clinical", "radiomics")}
    models.update({f"C3-{fm}": ("clinical", fm) for fm in FMS + ("pai", "hosny_score")})
    models.update({f"C4-{fm}": ("clinical", "radiomics", fm) for fm in ("colipri", "tangerine")})
    return models


def components(block: str) -> int | None:
    if block in ("clinical0", "clinical", "hosny_score"):
        return None
    return COMPONENTS


def fit_ladder_model(blocks: dict[str, np.ndarray], data: pd.DataFrame, inner: pd.DataFrame) -> dict:
    """run_lung1_ceiling_ladder.fit_fold on the whole cohort; block names map to clinical/radiomics/image."""
    time, event = data.os_time_days.to_numpy(float), data.os_event.to_numpy(int)
    roles = {name: ("clinical" if name.startswith("clinical") else "radiomics" if name == "radiomics" else "image")
             for name in blocks}

    def transform(fitted: dict, rows: np.ndarray) -> pd.DataFrame:
        parts, columns = [], []
        for name, block in fitted.items():
            values = block(blocks[name][rows])
            parts.append(values)
            columns += [f"{roles[name]}_{i:03d}" for i in range(values.shape[1])]
        return pd.DataFrame(np.hstack(parts), columns=columns)

    grid = list(itertools.product(ALPHAS, ALPHAS if len(blocks) > 1 else (0.0,)))
    scores = []
    for clinical_alpha, block_alpha in grid:
        fold_scores = []
        for inner_fold in range(3):
            valid = data.patient_id.isin(inner.loc[inner.inner_fold == inner_fold, "patient_id"]).to_numpy()
            train_rows, valid_rows = np.flatnonzero(~valid), np.flatnonzero(valid)
            fitted = {name: Block(x[train_rows], components(name)) for name, x in blocks.items()}
            model = fit_ladder(transform(fitted, train_rows), time[~valid], event[~valid], clinical_alpha, block_alpha)
            risk = model.predict_log_partial_hazard(transform(fitted, valid_rows)).to_numpy().reshape(-1)
            fold_scores.append(uno_c(time[~valid], event[~valid], time[valid], event[valid], risk))
        scores.append(float(np.mean(fold_scores)))
    clinical_alpha, block_alpha = grid[int(np.argmax(scores))]
    rows = np.arange(len(data))
    fitted = {name: Block(x, components(name)) for name, x in blocks.items()}
    model = fit_ladder(transform(fitted, rows), time, event, clinical_alpha, block_alpha)
    return {"kind": "ladder", "blocks": list(blocks), "roles": roles, "transforms": fitted, "cox": model,
            "clinical_alpha": float(clinical_alpha), "block_alpha": float(block_alpha),
            "inner_uno_c_36m": max(scores)}


def main() -> None:
    data = cohort(COHORT)
    time, event = data.os_time_days.to_numpy(float), data.os_event.to_numpy(int)
    inner = inner_assignments(data, FULL_FIT_FOLD, 0, 100)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blocks, provenance = {}, {}
    blocks["clinical0"], c0_columns = clinical_block(data, volume=False)
    blocks["clinical"], c1_columns = clinical_block(data, volume=True)
    blocks["radiomics"], radiomics_info = baseline_features(data, "radiomics", RADIOMICS)
    provenance["radiomics"] = {"sha256": radiomics_info["features_sha256"], "columns": radiomics_info["columns"]}
    for name in FMS + TUMOUR + ("hosny_score",):
        manifest = EMBEDDINGS / name / "original_ct" / "embedding_manifest.csv"
        _, blocks[name], manifest_sha, matrix_sha = load_embeddings(f"{name}={manifest}", data)
        provenance[name] = {"manifest_sha256": manifest_sha, "matrix_sha256": matrix_sha,
                            "dimension": int(blocks[name].shape[1])}
    models = {}
    for name, parts in ladder_models().items():
        models[name] = fit_ladder_model({b: blocks[b] for b in parts}, data, inner)
        print(f"{name}: alphas {models[name]['clinical_alpha']}/{models[name]['block_alpha']} "
              f"inner {models[name]['inner_uno_c_36m']:.4f}", flush=True)
    for name in FMS + TUMOUR:  # E3 frozen linear, PCA16
        alpha, scores = select_penalty(blocks[name], data, inner, COMPONENTS)
        models[name] = {"kind": "e3_linear", "block": name, "fitted": fit_cox(blocks[name], time, event, alpha, COMPONENTS),
                        "alpha": float(alpha), "inner_uno_c_36m": float(max(scores))}
        print(f"{name}: alpha {alpha} inner {max(scores):.4f}", flush=True)
    for name in ("stage_volume", "radiomics"):  # E3 baselines
        x, info = baseline_features(data, name, RADIOMICS)
        alpha, score = select_alpha(x, data, inner)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        train_x, _ = impute(x, x[:1])
        models[f"baseline_{name}"] = {"kind": "e3_baseline", "feature_set": name, "columns": info["columns"],
                                      "medians": medians, "fitted": fit_cox(train_x, time, event, alpha, 0),
                                      "alpha": float(alpha), "inner_uno_c_36m": float(score)}
        print(f"baseline_{name}: alpha {alpha} inner {score:.4f}", flush=True)
    models["hosny_zero_shot"] = {"kind": "zero_shot", "block": "hosny_score",
                                 "note": "risk = published-model log-odds of death at 2 years"}
    entries = {}
    for name, model in models.items():
        path = OUTPUT / f"{name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(model, stream)
        entries[name] = {"file": path.name, "sha256": sha256(path), "kind": model["kind"],
                         **{k: v for k, v in model.items() if k in ("clinical_alpha", "block_alpha", "alpha",
                                                                      "inner_uno_c_36m", "blocks", "block")}}
    manifest = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol": "docs/lung1_phase4_protocol.md", "training_cohort_sha256": sha256(COHORT),
                "n_training": len(data), "training_events": int(event.sum()),
                "inner_split": f"run_lung1_e3_frozen_linear.inner_assignments(fold={FULL_FIT_FOLD}, draw=0, budget=100)",
                "penalty_grid": list(ALPHAS), "image_components": COMPONENTS,
                "clinical_columns_C0": c0_columns, "clinical_columns_C1": c1_columns,
                "inputs": provenance, "models": entries,
                "radiogenomics_read": False}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256((OUTPUT / "manifest.json").read_bytes()).hexdigest()
    (OUTPUT / "FROZEN").write_text(f"manifest sha256 {digest}\n")
    print(f"Frozen {len(models)} models; manifest sha256 {digest}")


if __name__ == "__main__":
    main()
