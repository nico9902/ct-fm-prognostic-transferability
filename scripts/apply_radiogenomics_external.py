#!/usr/bin/env python3
"""Phase 4.4: apply the frozen LUNG1 models to NSCLC-Radiogenomics, once.

Pre-specified in docs/lung1_phase4_protocol.md. Two stages:

  --stage predict   checks the frozen-model hashes, builds every input from a copy of
                    the cohort WITHOUT outcome columns, writes the risk (linear predictor;
                    log-odds for Hosny zero-shot) and the predicted survival at 36/60 months,
                    and records the hash of the predictions;
  --stage evaluate  checks the hashes again, then reads the outcomes and computes:
                    Uno C at 36 months (censoring from the Radiogenomics cohort, fixed),
                    2,000 paired patient bootstrap resamples (seed 20261021), Holm families,
                    TOST +/-0.03 and minimum detectable difference; secondary Uno C at
                    60 months, Harrell C, calibration slope and observed/expected events at
                    36 months, and the Spearman correlation of internal vs external C.

Populations: primary = region_status ready (SEG, all models); secondary = global_status
ready (five FMs alone on all; C0 and COLIPRI - C0 on patients with pTNM).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats

from run_lung1_ceiling_ladder import FMS, clinical_block
from run_lung1_e3_baselines import features as baseline_features
from run_lung1_frozen_cox_e1 import ROOT, predict as predict_e3, sha256, uno_c
from test_lung1_bf16_selected_stats import holm

FROZEN = ROOT / "outputs/phase4_frozen_models"
COHORT = ROOT / "data/derived/radiogenomics/cohort_with_gtv_volume.csv"
EMBEDDINGS = ROOT / "outputs/radiogenomics_embeddings"
RADIOMICS = EMBEDDINGS / "radiomics_features.csv"
OUTPUT = ROOT / "outputs/phase4_external"
HORIZONS = (1095.75, 1826.25)
BOOTSTRAP, SEED, MARGIN = 2000, 20261021, 0.03
PRIMARY = [("colipri", "baseline_stage_volume"), ("C3-colipri", "C1"), ("colipri", "pai"), ("colipri", "hosny_zero_shot")]
OUTCOMES = ["os_time_days", "os_event"]


def check_frozen() -> dict:
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    expected = (FROZEN / "FROZEN").read_text().split()[-1]
    if hashlib.sha256((FROZEN / "manifest.json").read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen manifest changed")
    for name, entry in manifest["models"].items():
        if sha256(FROZEN / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Frozen model {name} changed")
    return manifest


def embeddings(name: str, data: pd.DataFrame) -> np.ndarray:
    manifest = pd.read_csv(EMBEDDINGS / name / "original_ct" / "embedding_manifest.csv", dtype=str).set_index("patient_id")
    if not set(data.patient_id) <= set(manifest.index):
        raise ValueError(f"{name}: embeddings missing for {sorted(set(data.patient_id) - set(manifest.index))[:5]}")
    manifest = manifest.loc[data.patient_id]
    if not np.array_equal(manifest.ct_series_uid.to_numpy(), data.ct_series_uid.to_numpy()):
        raise ValueError(f"{name}: CT series differ from the cohort")
    matrix = np.stack([np.load(path, allow_pickle=False) for path in manifest.embedding_path])
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name}: non-finite embeddings")
    return matrix


def block(name: str, data: pd.DataFrame) -> np.ndarray:
    if name == "clinical0":
        return clinical_block(data, volume=False)[0]
    if name == "clinical":
        return clinical_block(data, volume=True)[0]
    if name == "radiomics":
        return baseline_features(data, "radiomics", RADIOMICS)[0]
    return embeddings(name, data)


def apply(model: dict, data: pd.DataFrame, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    kind = model["kind"]
    if kind == "zero_shot":
        return embeddings("hosny_score", data)[:, 0], np.full((len(data), len(HORIZONS)), np.nan)
    if kind == "ladder":
        parts, columns = [], []
        for name in model["blocks"]:
            values = model["transforms"][name](block(name, data))
            parts.append(values)
            columns += [f"{model['roles'][name]}_{i:03d}" for i in range(values.shape[1])]
        x = pd.DataFrame(np.hstack(parts), columns=columns)
        risk = model["cox"].predict_log_partial_hazard(x).to_numpy().reshape(-1)
        survival = model["cox"].predict_survival_function(x, times=list(HORIZONS)).to_numpy().T
        return risk, survival
    if kind == "e3_linear":
        return predict_e3(model["fitted"], embeddings(model["block"], data), HORIZONS)
    if kind == "e3_baseline":
        x, info = baseline_features(data, model["feature_set"], RADIOMICS)
        if info["columns"] != model["columns"]:
            raise ValueError(f"{model['feature_set']}: feature columns differ from LUNG1")
        x = np.where(np.isfinite(x), x, model["medians"])
        return predict_e3(model["fitted"], x, HORIZONS)
    raise ValueError(kind)


def populations(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    secondary = data.loc[data.global_status.eq("ready")].reset_index(drop=True)
    return {"primary": data.loc[data.region_status.eq("ready")].reset_index(drop=True),
            "secondary": secondary,
            "secondary_ptnm": secondary.loc[secondary.t_stage.notna() & secondary.n_stage.notna()].reset_index(drop=True)}


MODELS_BY_POPULATION = {"secondary": list(FMS), "secondary_ptnm": ["C0", "colipri"]}


def stage_predict() -> None:
    manifest = check_frozen()
    data = pd.read_csv(COHORT, dtype={"patient_id": str, "ct_series_uid": str}).drop(columns=OUTCOMES)
    rows = []
    for population, subset in populations(data).items():
        names = MODELS_BY_POPULATION.get(population, list(manifest["models"]))
        for name in names:
            with (FROZEN / manifest["models"][name]["file"]).open("rb") as stream:
                model = pickle.load(stream)
            risk, survival = apply(model, subset, manifest)
            if not np.isfinite(risk).all():
                raise ValueError(f"{population}/{name}: non-finite risk")
            rows += [{"population": population, "model": name, "patient_id": pid, "risk": float(r),
                      "survival_36m": float(s[0]), "survival_60m": float(s[1])}
                     for pid, r, s in zip(subset.patient_id, risk, survival)]
        print(f"{population}: {len(subset)} patients, {len(names)} models", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (OUTPUT / "PREDICTED").write_text(json.dumps({"predictions_sha256": sha256(path),
                                                  "frozen_manifest_sha256": sha256(FROZEN / "manifest.json"),
                                                  "cohort_sha256": sha256(COHORT),
                                                  "outcomes_read": False}, indent=2) + "\n")
    print(f"Predictions written: {path}")


def wald(difference: float, draws: np.ndarray) -> dict:
    se = float(np.nanstd(draws, ddof=1))
    lo90, hi90 = np.nanpercentile(draws, [5, 95])
    p_lower = float(stats.norm.sf((difference + MARGIN) / se))
    p_upper = float(stats.norm.cdf((difference - MARGIN) / se))
    return {"difference": difference, "se": se, "ci95": [float(np.nanpercentile(draws, 2.5)),
                                                        float(np.nanpercentile(draws, 97.5))],
            "p": float(2 * stats.norm.sf(abs(difference / se))), "p_tost": max(p_lower, p_upper),
            "equivalent_within_margin": bool(-MARGIN < lo90 and hi90 < MARGIN),
            "minimum_detectable_difference": float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se)}


def family(tests: dict) -> dict:
    keys = list(tests)
    for key, p in zip(keys, holm([tests[k]["p"] for k in keys])):
        tests[key]["p_holm"] = p
    return tests


def calibration(time, event, risk, survival_36m) -> dict:
    frame = pd.DataFrame({"t": time, "e": event, "lp": risk})
    slope = CoxPHFitter().fit(frame, "t", "e").params_["lp"]
    km = KaplanMeierFitter().fit(time, event)
    observed = 1.0 - float(km.predict(HORIZONS[0]))
    expected = float(np.mean(1.0 - survival_36m)) if np.isfinite(survival_36m).all() else np.nan
    return {"calibration_slope": float(slope), "observed_risk_36m": observed, "expected_risk_36m": expected,
            "observed_expected_ratio_36m": observed / expected if np.isfinite(expected) else None}


def internal_c() -> dict[str, float]:
    ladder = pd.read_csv(ROOT / "outputs/lung1_ceiling_ladder/primary_tumour_centred/summary.csv").set_index("model").uno_c_36m
    base = pd.read_csv(ROOT / "outputs/lung1_e3_baselines/summary.csv")
    base = base.loc[base.budget_pct == 100].set_index("model").uno_c_36m
    out = ladder.to_dict()
    out.update({f"baseline_{k}": v for k, v in base.items()})
    return out


def stage_evaluate() -> None:
    check_frozen()
    record = json.loads((OUTPUT / "PREDICTED").read_text())
    if sha256(OUTPUT / "predictions.csv") != record["predictions_sha256"]:
        raise ValueError("Predictions changed after the predict stage")
    if sha256(FROZEN / "manifest.json") != record["frozen_manifest_sha256"]:
        raise ValueError("Frozen models changed after the predict stage")
    predictions = pd.read_csv(OUTPUT / "predictions.csv", dtype={"patient_id": str})
    data = pd.read_csv(COHORT, dtype={"patient_id": str}).set_index("patient_id")
    rng = np.random.default_rng(SEED)
    result = {"protocol": "docs/lung1_phase4_protocol.md", "bootstrap": BOOTSTRAP, "seed": SEED,
              "censoring": "Kaplan-Meier of the Radiogenomics population, fixed across resamples",
              "predicted_record": record, "populations": {}}
    internal = internal_c()
    for population, frame in predictions.groupby("population", sort=False):
        wide = frame.pivot(index="patient_id", columns="model", values="risk")
        surv36 = frame.pivot(index="patient_id", columns="model", values="survival_36m")
        ids = wide.index.to_numpy()
        t = data.loc[ids, "os_time_days"].to_numpy(float)
        e = data.loc[ids, "os_event"].to_numpy(int)
        draws = [rng.integers(0, len(ids), len(ids)) for _ in range(BOOTSTRAP)]
        point, boot, models = {}, {}, {}
        for name in wide.columns:
            risk = wide[name].to_numpy(float)
            point[name] = uno_c(t, e, t, e, risk)
            boot[name] = np.array([uno_c(t, e, t[d], e[d], risk[d]) for d in draws])
            models[name] = {"uno_c_36m": point[name],
                            "ci95": [float(np.nanpercentile(boot[name], 2.5)), float(np.nanpercentile(boot[name], 97.5))],
                            "uno_c_60m": uno_c(t, e, t, e, risk, tau=HORIZONS[1]),
                            "harrell_c": float(concordance_index(t, -risk, e)),
                            "internal_lung1_uno_c_36m": internal.get(name),
                            **calibration(t, e, risk, surv36[name].to_numpy(float))}
        out = {"n": int(len(ids)), "events": int(e.sum()),
               "events_by_36m": int(((e == 1) & (t <= HORIZONS[0])).sum()),
               "median_follow_up_days": float(np.median(t)), "models": models,
               "chance": family({name: wald(point[name] - 0.5, boot[name] - 0.5) for name in wide.columns})}
        if population == "primary":
            out["primary_contrasts"] = family({f"{a} - {b}": wald(point[a] - point[b], boot[a] - boot[b])
                                               for a, b in PRIMARY})
            others = [n for n in wide.columns if n != "colipri"]
            out["secondary_contrasts_vs_colipri"] = {f"colipri - {n}": wald(point["colipri"] - point[n],
                                                                           boot["colipri"] - boot[n]) for n in others}
            pairs = [(internal[n], point[n]) for n in wide.columns if internal.get(n) is not None]
            rho = stats.spearmanr(*zip(*pairs))
            out["ranking_transport"] = {"n_models": len(pairs), "spearman_internal_vs_external": float(rho.statistic),
                                        "p": float(rho.pvalue)}
        if population == "secondary_ptnm":
            out["contrasts"] = {"colipri - C0": wald(point["colipri"] - point["C0"], boot["colipri"] - boot["C0"])}
        result["populations"][population] = out
        print(f"== {population}: n={len(ids)} events={int(e.sum())}")
        for name in sorted(models, key=lambda n: -models[n]["uno_c_36m"]):
            m = models[name]
            print(f"  {name:24s} {m['uno_c_36m']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  internal "
                  f"{m['internal_lung1_uno_c_36m'] if m['internal_lung1_uno_c_36m'] is None else round(m['internal_lung1_uno_c_36m'], 3)}")
    (OUTPUT / "external_validation.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    primary = result["populations"]["primary"]
    for key, value in primary["primary_contrasts"].items():
        print(f"PRIMARY {key:40s} {value['difference']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] "
              f"p_holm={value['p_holm']:.3f} MDD={value['minimum_detectable_difference']:.3f}")
    print("ranking transport:", primary["ranking_transport"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "evaluate"), required=True)
    args = parser.parse_args()
    stage_predict() if args.stage == "predict" else stage_evaluate()


if __name__ == "__main__":
    main()
