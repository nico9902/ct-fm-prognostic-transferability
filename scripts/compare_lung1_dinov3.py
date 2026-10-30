#!/usr/bin/env python3
"""Paired pooled Uno C contrasts of DINOv3 against the 3-D FMs and baselines.

Same estimator, draws and seed as `run_lung1_bf16_selected_baselines.py`:
500 patient bootstrap resamples of the 416 out-of-fold predictions with the
censoring curve fixed to the full cohort. Logged as one W&B summary run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_lung1_bf16_selected_baselines import (  # noqa: E402
    BOOTSTRAP_SEED, COHORT, FM_METHODS, OUTPUT, SELECTED, pooled_risk,
)
from run_lung1_e2_encoder_bf16_lightning import write_json  # noqa: E402
from run_lung1_e3_baselines import BASELINES  # noqa: E402
from run_lung1_frozen_cox_e1 import cohort, uno_c  # noqa: E402

OTHERS = ("tangerine", "radjepa", "colipri", "ctclip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--wandb-project", default="lung1-prognostic-transferability")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    args = parser.parse_args()
    data = cohort(COHORT)
    t = data.os_time_days.to_numpy(dtype=float)
    e = data.os_event.to_numpy(dtype=int)
    risks = {f"baseline/{name}": pooled_risk(OUTPUT / name / "out_of_fold_predictions.csv", data)
             for name in BASELINES}
    for method in FM_METHODS:
        for model in (*OTHERS, "dinov3"):
            risks[f"{method}/{model}"] = pooled_risk(
                SELECTED / method / model / "out_of_fold_predictions.csv", data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(data), len(data)) for _ in range(args.bootstrap)]
    point = {key: float(uno_c(t, e, t, e, value)) for key, value in risks.items()}
    boot = {key: np.array([uno_c(t, e, t[d], e[d], value[d]) for d in draws])
            for key, value in risks.items()}

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    contrasts = {}
    for method in FM_METHODS:
        mine = f"{method}/dinov3"
        for other in (*(f"{method}/{model}" for model in OTHERS),
                      *(f"baseline/{name}" for name in BASELINES)):
            contrasts[f"{mine} - {other}"] = {"difference": point[mine] - point[other],
                                              "ci95": interval(boot[mine] - boot[other])}
        if method != "linear":
            contrasts[f"{mine} - linear/dinov3"] = {
                "difference": point[mine] - point["linear/dinov3"],
                "ci95": interval(boot[mine] - boot["linear/dinov3"])}
    result = {"metric": "pooled_uno_c_36m", "risk": "negative_log_predicted_survival_at_36_months",
              "bootstrap": args.bootstrap, "seed": BOOTSTRAP_SEED,
              "note": "conditional on fitted fold models; no multiplicity adjustment",
              "conditions": {key: {"uno_c_36m": point[key], "ci95": interval(boot[key])}
                             for key in risks},
              "contrasts": contrasts}
    write_json(SELECTED / "dinov3_paired_uno_c.json", result)
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name="5CV-BF16-dinov3-paired-comparison",
                         group="lung1-5cv-bf16-dinov3-comparison", job_type="paired-comparison",
                         mode=args.wandb_mode, dir=str(ROOT / "logs/lung1_bf16_selected"),
                         config={"bootstrap": args.bootstrap, "seed": BOOTSTRAP_SEED})
        summary = {f"pooled_uno_c/{key}": value["uno_c_36m"]
                   for key, value in result["conditions"].items()}
        for key, value in contrasts.items():
            summary[f"contrast/{key}"] = value["difference"]
            summary[f"contrast/{key}/ci_low"], summary[f"contrast/{key}/ci_high"] = value["ci95"]
        run.summary.update(summary)
        run.finish()
    for key, value in contrasts.items():
        print(f"{key}: {value['difference']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
