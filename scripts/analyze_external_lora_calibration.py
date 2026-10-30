#!/usr/bin/env python3
"""Exploratory external LoRA: test against chance and calibration in the primary Radiogenomics population.

Completes Table 8 for the LoRA rows with the quantities reported for the frozen models
(apply_radiogenomics_external.py): Wald test of Uno C at 36 months against 0.5 with the bootstrap
standard error, calibration slope and mean predicted 36-month mortality. The bootstrap repeats the
resamples of run_lung1_e5_lora.py (seed 20261023, 2,000 draws), so the intervals match summary.json.
As stated in the Methods, the external LoRA analysis is not adjusted for multiplicity.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from apply_radiogenomics_external import calibration
from run_lung1_frozen_cox_e1 import ROOT, uno_c

EXTERNAL = ROOT / "data/derived/radiogenomics/cohort_with_gtv_volume.csv"
FROZEN = ROOT / "outputs/phase4_external/predictions.csv"
OUTPUT = ROOT / "outputs/lung1_e5_lora_external_calibration.json"
RUNS = {"colipri": "lung1_e5_lora_v100_fp16", "tangerine": "lung1_e5_lora_v100_fp16",
        "radjepa": "lung1_e5_lora_a100_bf16", "ctclip": "lung1_e5_lora_a100_bf16", "dinov3": "lung1_e5_lora_a100_bf16"}
TAU, BOOTSTRAP, SEED = 1095.75, 2000, 20261023


def main() -> None:
    data = pd.read_csv(EXTERNAL, dtype={"patient_id": str})
    data = data.loc[data.global_status.eq("ready") & data.region_status.eq("ready")]
    frozen = pd.read_csv(FROZEN, dtype={"patient_id": str})
    out = {}
    for model, run in RUNS.items():
        folder = ROOT / "outputs" / run / model
        part = data.merge(pd.read_csv(folder / "predictions.csv", dtype={"patient_id": str}),
                          on="patient_id", validate="one_to_one")
        comparison = frozen.loc[(frozen.model == model) & (frozen.population == "primary")]
        part = part.merge(comparison[["patient_id", "risk"]].rename(columns={"risk": "frozen_risk"}),
                          on="patient_id", validate="one_to_one")
        t, e = part.os_time_days.to_numpy(float), part.os_event.to_numpy(int)
        r, f = part.risk.to_numpy(float), part.frozen_risk.to_numpy(float)

        def score(ix, risk):
            return float(uno_c(t[ix], e[ix], t[ix], e[ix], risk[ix], TAU))

        point = score(np.arange(len(part)), r)
        rng, draws = np.random.default_rng(SEED), []
        for _ in range(BOOTSTRAP):  # same resamples and validity rule as run_lung1_e5_lora.evaluate
            sample = rng.integers(0, len(part), len(part))
            a, b = score(sample, r), score(sample, f)
            if np.isfinite(a) and np.isfinite(b):
                draws.append(a)
        se = float(np.std(draws, ddof=1))
        summary = json.load(open(folder / "summary.json"))["populations"]["primary"]
        if abs(point - summary["uno_c_36m"]) > 1e-9:
            raise ValueError(f"{model}: Uno C differs from summary.json")
        out[model] = {"patients": len(part), "uno_c_36m": point,
                      "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "se": se,
                      "p_vs_chance": float(2 * stats.norm.sf(abs((point - 0.5) / se))),
                      "risk_sd": float(np.std(r)),
                      **calibration(t, e, r, part.survival_11.to_numpy(float))}
        m = out[model]
        print(f"{model:10s} C={point:.3f} p={m['p_vs_chance']:.3f} slope={m['calibration_slope']:.2f} "
              f"predicted={m['expected_risk_36m']:.2f} observed={m['observed_risk_36m']:.2f}")
    OUTPUT.write_text(json.dumps({"population": "primary", "bootstrap": BOOTSTRAP, "seed": SEED,
                                  "multiplicity": "none (exploratory)", "models": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
