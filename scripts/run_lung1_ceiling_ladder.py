#!/usr/bin/env python3
"""LUNG1 empirical ceiling ladder C0-C4 (phase 1.1, 1.2, 1.5).

Locked split: five outer folds of repetition zero, three event-stratified inner
folds (E3 inner seeds) inside each outer development set.

Blocks (all transforms fitted on training patients only)
* clinical: age (+ missing indicator), male sex, T2/T3/T4 (T5 merged into T4),
            N1/N2/N3 (N4 merged into N3), stage II/IIIa/IIIb/missing, histology
            large cell / adenocarcinoma / NOS / missing (squamous reference).
            M is excluded (411 of 416 are M0).
* volume:   log1p GTV volume in cm3 (part of the clinical block from C1 on).
* radiomics: 107 PyRadiomics Original GTV features, median imputation,
            standardization, PCA16 (or all features with --radiomics-full).
* image:    frozen E1 global embedding, standardization, PCA16.

Models
  C0 = clinical, C1 = C0 + volume, C2 = C1 + radiomics,
  C3-<fm> = C1 + image (five FMs), C4-<fm> = C1 + radiomics + image.
C4 uses COLIPRI as the pre-specified primary image block; TANGERINE is a sensitivity.

Penalties: lifelines L2 Cox with one penalty for the clinical block and one shared
penalty for the radiomics/image blocks, both chosen on the 3 x 3 grid by mean inner
Uno C at 36 months.

Statistics: mean outer-fold Uno C at 36 months; paired held-out patient bootstrap
within each fold; two-sided Wald test with bootstrap SE, Holm within each family;
TOST equivalence with margin 0.03 (90% percentile interval within the margin);
minimum detectable difference (80% power, two-sided 5%) from the paired SE.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_e3_baselines import features as baseline_features
from run_lung1_e3_frozen_linear import inner_assignments, write_json
from run_lung1_frozen_cox_e1 import (
    ALPHAS, COHORT, FOLDS, GRID, ROOT, SEED, cohort, fold_metrics, load_embeddings,
    locked_folds, sha256, uno_c,
)
from test_lung1_bf16_selected_stats import holm

FMS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
TUMOR = ("pai", "hosny", "hosny_score")  # phase 2.5: tumour-centred image blocks
RADIOMICS = ROOT / "outputs/lung1_e3_baselines/radiomics_features.csv"
OUTPUT = ROOT / "outputs/lung1_ceiling_ladder"
E3_FM = ROOT / "outputs/lung1_e3_frozen_linear_pca16"
BOOT_SEED = 20261019
MARGIN = 0.03
COMPONENTS = 16


def clinical_block(data: pd.DataFrame, volume: bool) -> tuple[np.ndarray, list[str]]:
    age = pd.to_numeric(data.age, errors="coerce").to_numpy(float)
    t = pd.to_numeric(data.t_stage, errors="coerce").clip(upper=4)
    n = pd.to_numeric(data.n_stage, errors="coerce").clip(upper=3)
    stage = data.overall_stage.astype(str).str.lower()
    histology = data.histology.astype(str).str.lower()
    if not stage.isin(("i", "ii", "iiia", "iiib", "nan")).all():
        raise ValueError("Unexpected stage label")
    if not histology.isin(("squamous cell carcinoma", "large cell", "adenocarcinoma", "nos", "nan")).all():
        raise ValueError("Unexpected histology label")
    if t.isna().any() or n.isna().any():
        raise ValueError("Missing T or N")
    columns = {
        "age": age,  # NaN imputed with the training median
        "age_missing": np.isnan(age).astype(float),
        "male": (data.sex.astype(str).str.lower() == "male").to_numpy(float),
        **{f"T{k}": (t == k).to_numpy(float) for k in (2, 3, 4)},
        **{f"N{k}": (n == k).to_numpy(float) for k in (1, 2, 3)},
        **{f"stage_{k}": (stage == k).to_numpy(float) for k in ("ii", "iiia", "iiib", "nan")},
        **{f"hist_{k.split()[0]}": (histology == k).to_numpy(float)
           for k in ("large cell", "adenocarcinoma", "nos", "nan")},
    }
    if volume:
        columns["log1p_gtv_volume"] = np.log1p(pd.to_numeric(data.gtv_volume_cm3, errors="raise").to_numpy(float))
    return np.column_stack(list(columns.values())), list(columns)


class Block:
    """Median imputation, standardization and optional PCA, fitted on training rows."""

    def __init__(self, x: np.ndarray, components: int | None):
        with np.errstate(all="ignore"):
            self.median = np.nan_to_num(np.nanmedian(x, axis=0))
        x = self._impute(x)
        self.keep = np.std(x, axis=0) > 0
        self.scaler = StandardScaler().fit(x[:, self.keep])
        self.pca = None
        if components:
            self.pca = PCA(n_components=components, svd_solver="randomized",
                           random_state=SEED).fit(self.scaler.transform(x[:, self.keep]))

    def _impute(self, x):
        return np.where(np.isfinite(x), x, self.median)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(self._impute(x)[:, self.keep])
        return self.pca.transform(z) if self.pca is not None else z


def design(blocks: dict[str, np.ndarray], rows: np.ndarray, components: dict[str, int | None]):
    fitted = {name: Block(x[rows], components[name]) for name, x in blocks.items()}

    def transform(subset: np.ndarray) -> pd.DataFrame:
        parts, columns = [], []
        for name, block in fitted.items():
            values = block(blocks[name][subset])
            parts.append(values)
            columns += [f"{name}_{i:03d}" for i in range(values.shape[1])]
        return pd.DataFrame(np.hstack(parts), columns=columns)
    return transform


def fit(x: pd.DataFrame, time, event, clinical_alpha: float, block_alpha: float) -> CoxPHFitter:
    penalty = np.array([clinical_alpha if c.startswith("clinical_") else block_alpha for c in x.columns])
    return CoxPHFitter(penalizer=penalty, l1_ratio=0.0).fit(
        x.assign(time=time, event=event), duration_col="time", event_col="event", show_progress=False)


def fit_fold(task: tuple) -> dict:
    name, blocks, components, data, test_ids, fold, last_death = task
    test = data.patient_id.isin(test_ids).to_numpy()
    dev_rows = np.flatnonzero(~test)
    train = data.iloc[dev_rows].reset_index(drop=True)
    time, event = train.os_time_days.to_numpy(float), train.os_event.to_numpy(int)
    inner = inner_assignments(train, fold, 0, 100)
    grid = list(itertools.product(ALPHAS, ALPHAS if len(blocks) > 1 else (0.0,)))
    scores = []
    for clinical_alpha, block_alpha in grid:
        inner_scores = []
        for inner_fold in range(3):
            valid = train.patient_id.isin(inner.loc[inner.inner_fold == inner_fold, "patient_id"]).to_numpy()
            transform = design(blocks, dev_rows[~valid], components)
            model = fit(transform(dev_rows[~valid]), time[~valid], event[~valid], clinical_alpha, block_alpha)
            risk = model.predict_log_partial_hazard(transform(dev_rows[valid])).to_numpy().reshape(-1)
            score = uno_c(time[~valid], event[~valid], time[valid], event[valid], risk)
            if not np.isfinite(score):
                raise ValueError("Undefined inner Uno C")
            inner_scores.append(score)
        scores.append(float(np.mean(inner_scores)))
    clinical_alpha, block_alpha = grid[int(np.argmax(scores))]
    transform = design(blocks, dev_rows, components)
    model = fit(transform(dev_rows), time, event, clinical_alpha, block_alpha)
    x_test = transform(np.flatnonzero(test))
    risk = model.predict_log_partial_hazard(x_test).to_numpy().reshape(-1)
    survival = model.predict_survival_function(x_test, times=GRID).to_numpy().T
    held = data.loc[test]
    t_time, t_event = held.os_time_days.to_numpy(float), held.os_event.to_numpy(int)
    metrics = fold_metrics(time, event, t_time, t_event, risk, survival)
    metrics["harrell_c_full"] = float(concordance_index(t_time, -risk, t_event))
    metrics["uno_c_full"] = uno_c(time, event, t_time, t_event, risk, tau=last_death)
    coef = model.params_
    return {"model": name, "fold": fold, "clinical_alpha": float(clinical_alpha),
            "block_alpha": float(block_alpha), "inner_uno_c_36m": max(scores),
            "n_train": len(dev_rows), "n_test": int(test.sum()),
            "coefficient_l2_by_block": {b: float(np.linalg.norm([v for k, v in coef.items()
                                                                 if k.startswith(f"{b}_")])) for b in blocks},
            "metrics": {k: float(v) for k, v in metrics.items()},
            "predictions": [{"patient_id": str(p), "os_time_days": float(a), "os_event": int(b), "risk": float(r)}
                            for p, a, b, r in zip(held.patient_id, t_time, t_event, risk)]}


def contrast_rows(family: str, pairs, boot, point) -> list[dict]:
    rows = []
    for first, second in pairs:
        draws = boot[first] - boot[second]
        difference = point[first] - point[second]
        se = float(np.std(draws, ddof=1))
        p_lower = float(stats.norm.sf((difference + MARGIN) / se))  # H0: diff <= -margin
        p_upper = float(stats.norm.cdf((difference - MARGIN) / se))  # H0: diff >= +margin
        lo90, hi90 = np.percentile(draws, [5, 95])
        rows.append({"family": family, "contrast": f"{first} - {second}", "difference": difference,
                     "se": se, "ci95_lower": float(np.percentile(draws, 2.5)),
                     "ci95_upper": float(np.percentile(draws, 97.5)),
                     "p": float(2 * stats.norm.sf(abs(difference / se))),
                     "ci90_lower": float(lo90), "ci90_upper": float(hi90),
                     "p_tost": max(p_lower, p_upper),
                     "equivalent_within_margin": bool(-MARGIN < lo90 and hi90 < MARGIN),
                     "minimum_detectable_difference": float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se)})
    for row, adjusted, adjusted_tost in zip(rows, holm([r["p"] for r in rows]), holm([r["p_tost"] for r in rows])):
        row["p_holm"] = adjusted
        row["p_tost_holm"] = adjusted_tost
    return rows


def zero_shot_fold(data: pd.DataFrame, folds: pd.DataFrame, score: np.ndarray, fold: int,
                   last_death: float) -> dict:
    ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
    test = data.patient_id.isin(ids).to_numpy()
    time, event = data.os_time_days.to_numpy(float), data.os_event.to_numpy(int)
    risk = score[test]
    metrics = {"uno_c_36m": uno_c(time[~test], event[~test], time[test], event[test], risk),
               "uno_c_full": uno_c(time[~test], event[~test], time[test], event[test], risk, tau=last_death),
               "harrell_c_full": float(concordance_index(time[test], -risk, event[test])),
               "ibs_3m_36m": float("nan")}  # no survival curve without a fitted baseline
    return {"model": "hosny_zero_shot", "fold": fold, "metrics": {k: float(v) for k, v in metrics.items()},
            "predictions": [{"patient_id": str(p), "os_time_days": float(a), "os_event": int(b), "risk": float(r)}
                            for p, a, b, r in zip(data.patient_id[test], time[test], event[test], risk)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--radiomics-full", action="store_true",
                        help="Sensitivity: all 107 radiomics features instead of PCA16")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tumor-centred", action="store_true",
                        help="Add C3-pai, C3-hosny (PCA16) and C3-hosny_score (1-d zero-shot score)")
    args = parser.parse_args()
    data = cohort(COHORT)
    folds = locked_folds(data, FOLDS)
    output = OUTPUT / (("radiomics_full" if args.radiomics_full else "primary")
                      + ("_tumour_centred" if args.tumor_centred else ""))
    output.mkdir(parents=True, exist_ok=True)
    c0, c0_columns = clinical_block(data, volume=False)
    c1, c1_columns = clinical_block(data, volume=True)
    radiomics, radiomics_info = baseline_features(data, "radiomics", RADIOMICS)
    images = {}
    for fm in FMS + (TUMOR if args.tumor_centred else ()):
        manifest = ROOT / "outputs/lung1_e1_embeddings" / fm / "original_ct" / "embedding_manifest.csv"
        _, images[fm], _, _ = load_embeddings(f"{fm}={manifest}", data)
    rad_components = None if args.radiomics_full else COMPONENTS
    models = {"C0": {"clinical": c0}, "C1": {"clinical": c1}, "C2": {"clinical": c1, "radiomics": radiomics}}
    for fm in FMS:
        models[f"C3-{fm}"] = {"clinical": c1, "image": images[fm]}
    for fm in ("colipri", "tangerine"):
        models[f"C4-{fm}"] = {"clinical": c1, "radiomics": radiomics, "image": images[fm]}
    if args.tumor_centred:
        for name in TUMOR:
            models[f"C3-{name}"] = {"clinical": c1, "image": images[name]}
    components = {"clinical": None, "radiomics": rad_components, "image": COMPONENTS}
    write_json(output / "protocol.json", {
        "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS), "outer_repetition": 0,
        "inner": "3fold_event_stratified (E3 inner seeds)", "penalties": list(ALPHAS),
        "penalty_selection": "clinical-block penalty x shared radiomics/image penalty, 3x3 grid, mean inner Uno C 36m",
        "clinical_columns_C0": c0_columns, "clinical_columns_C1": c1_columns,
        "coding": "T5 merged into T4, N4 merged into N3, M excluded, age median-imputed + indicator",
        "radiomics": {"sha256": radiomics_info["features_sha256"], "n_features": radiomics_info["n_features"],
                      "components": rad_components or "all"},
        "image_components": COMPONENTS, "c4_primary_image_block": "colipri", "models": list(models),
        "tost_margin": MARGIN, "bootstrap": args.bootstrap, "bootstrap_seed": BOOT_SEED})
    last_death = float(data.loc[data.os_event == 1, "os_time_days"].max())
    tasks = []
    for name, blocks in models.items():
        for fold in range(5):
            if not (output / "fits" / name / f"fold_{fold}.json").exists():
                ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
                parts = {b: components[b] for b in blocks}
                if name == "C3-hosny_score":
                    parts["image"] = None  # a single standardized score, no PCA
                tasks.append((name, blocks, parts, data, ids, fold, last_death))
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(fit_fold, tasks):
            write_json(output / "fits" / result["model"] / f"fold_{result['fold']}.json", result)
            print(f"{result['model']} fold={result['fold']} Uno36={result['metrics']['uno_c_36m']:.4f}", flush=True)
    fits = {name: [json.loads((output / "fits" / name / f"fold_{f}.json").read_text()) for f in range(5)]
            for name in models}
    for fm in FMS + (("pai", "hosny") if args.tumor_centred else ()):  # FM alone: E3 PCA16 full-budget fits
        fits[fm] = [json.loads((E3_FM / "fits" / fm / f"fold_{f}" / "draw_0_budget_100.json").read_text())
                    for f in range(5)]
    if args.tumor_centred:  # Hosny zero-shot: published log-odds of death used as the risk, nothing fitted
        fits["hosny_zero_shot"] = [zero_shot_fold(data, folds, images["hosny_score"][:, 0], fold, last_death)
                                   for fold in range(5)]
    rng = np.random.default_rng(BOOT_SEED)
    counts, development = {}, {}
    for fold in range(5):
        ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
        test, dev = data.loc[data.patient_id.isin(ids)], data.loc[~data.patient_id.isin(ids)]
        sampled = rng.integers(0, len(test), size=(args.bootstrap, len(test)))
        counts[fold] = np.stack([np.bincount(row, minlength=len(test)) for row in sampled])
        development[fold] = (dev.os_time_days.to_numpy(float), dev.os_event.to_numpy(int))
    boot, point, summary = {}, {}, []
    for name, results in fits.items():
        per_fold = []
        for fold, result in enumerate(results):
            if [p["patient_id"] for p in result["predictions"]] != [p["patient_id"] for p in fits["C0"][fold]["predictions"]]:
                raise ValueError(f"{name}: held-out patient order differs in fold {fold}")
            numerator, denominator = weighted_pair_matrices(result["predictions"], *development[fold])
            per_fold.append(bootstrap_scores(numerator, denominator, counts[fold]))
        boot[name] = np.mean(per_fold, axis=0)
        point[name] = float(np.mean([r["metrics"]["uno_c_36m"] for r in results]))
        summary.append({"model": name, "uno_c_36m": point[name],
                        "ci95_lower": float(np.percentile(boot[name], 2.5)),
                        "ci95_upper": float(np.percentile(boot[name], 97.5)),
                        "fold_min": float(min(r["metrics"]["uno_c_36m"] for r in results)),
                        "fold_max": float(max(r["metrics"]["uno_c_36m"] for r in results)),
                        "harrell_c_full": float(np.mean([r["metrics"]["harrell_c_full"] for r in results])),
                        "ibs_3m_36m": float(np.mean([r["metrics"]["ibs_3m_36m"] for r in results])),
                        "alphas": [(r.get("clinical_alpha"), r.get("block_alpha"), r.get("alpha")) for r in results]})
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    families = {
        "ladder_primary": [("C1", "C0"), ("C2", "C1"), ("C3-colipri", "C1"), ("C4-colipri", "C3-colipri"),
                           ("C4-colipri", "C2")],
        "image_over_C1": [(f"C3-{fm}", "C1") for fm in FMS],
        "C1_over_image": [(f"C3-{fm}", fm) for fm in FMS],
        "radiomics_over_C3": [(f"C4-{fm}", f"C3-{fm}") for fm in ("colipri", "tangerine")],
        "image_vs_radiomics": [(f"C3-{fm}", "C2") for fm in FMS],
    }
    if args.tumor_centred:
        families["tumour_centred_over_C1"] = [(f"C3-{t}", "C1") for t in TUMOR]
        families["tumour_centred_vs_radiomics"] = [(f"C3-{t}", "C2") for t in TUMOR]
        families["tumour_vs_whole_ct"] = [("C3-pai", "C3-colipri"), ("C3-hosny", "C3-colipri"),
                                          ("C3-hosny_score", "C3-colipri"), ("pai", "colipri"), ("hosny", "colipri"),
                                          ("hosny_zero_shot", "colipri")]
    tests = [row for family, pairs in families.items() for row in contrast_rows(family, pairs, boot, point)]
    pd.DataFrame(tests).to_csv(output / "tests.csv", index=False)
    print(pd.DataFrame(summary).drop(columns="alphas").round(4).to_string(index=False))
    print(pd.DataFrame(tests)[["family", "contrast", "difference", "ci95_lower", "ci95_upper", "p_holm",
                               "equivalent_within_margin", "minimum_detectable_difference"]].round(4).to_string(index=False))
    (output / "COMPLETE").write_text("done\n")


if __name__ == "__main__":
    main()
