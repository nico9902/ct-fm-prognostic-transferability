#!/usr/bin/env python3
"""Phase 2.3 statistics: tumour-centred methods (Pai, Hosny) on the selected-checkpoint protocol.

Same estimator, draws and seed as test_lung1_bf16_selected_stats.py: pooled Uno C at
36 months on the 416 out-of-fold predictions, 500 paired patient bootstrap resamples,
censoring curve fixed to the full cohort; two-sided Wald z with bootstrap SE, Holm
within each family. Hosny zero-shot uses the published-model log-odds of death at
2 years as the risk (nothing fitted).

Families
  chance        each tumour-centred condition versus 0.5
  vs_colipri    each condition versus COLIPRI on the same rung (fine-tuned and
                zero-shot Hosny versus frozen linear COLIPRI)
  baselines     each condition versus stage + volume and versus radiomics
  hosny_rungs   fine-tuned - zero-shot, fine-tuned - linear, MLP - linear (Hosny);
                MLP - linear (Pai)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_lung1_bf16_selected_baselines import BOOTSTRAP_SEED, COHORT, OUTPUT, SELECTED, pooled_risk  # noqa: E402
from run_lung1_e2_encoder_bf16_lightning import write_json  # noqa: E402
from run_lung1_e3_baselines import BASELINES  # noqa: E402
from run_lung1_frozen_cox_e1 import cohort, uno_c  # noqa: E402
from test_lung1_bf16_selected_stats import family, wald  # noqa: E402

ZERO_SHOT = ROOT / "outputs/lung1_e1_embeddings/hosny/zero_shot_scores.csv"
CONDITIONS = ("linear/pai", "mlp/pai", "linear/hosny", "mlp/hosny", "finetune/hosny", "zeroshot/hosny")
REFERENCE = {"linear/pai": "linear/colipri", "mlp/pai": "mlp/colipri", "linear/hosny": "linear/colipri",
             "mlp/hosny": "mlp/colipri", "finetune/hosny": "linear/colipri", "zeroshot/hosny": "linear/colipri"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    data = cohort(COHORT)
    t = data.os_time_days.to_numpy(dtype=float)
    e = data.os_event.to_numpy(dtype=int)
    risks = {f"baseline/{name}": pooled_risk(OUTPUT / name / "out_of_fold_predictions.csv", data)
             for name in BASELINES}
    for key in (*CONDITIONS[:-1], "linear/colipri", "mlp/colipri"):
        risks[key] = pooled_risk(SELECTED / key / "out_of_fold_predictions.csv", data)
    zero = pd.read_csv(ZERO_SHOT, dtype={"patient_id": str}).set_index("patient_id")
    risks["zeroshot/hosny"] = zero.loc[data.patient_id, "logit_dead"].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(data), len(data)) for _ in range(args.bootstrap)]
    point = {key: float(uno_c(t, e, t, e, value)) for key, value in risks.items()}
    boot = {key: np.array([uno_c(t, e, t[d], e[d], value[d]) for d in draws]) for key, value in risks.items()}

    def contrast(a: str, b: str) -> dict:
        return wald(point[a] - point[b], boot[a] - boot[b])

    result = {
        "metric": "pooled_uno_c_36m", "bootstrap": args.bootstrap, "seed": BOOTSTRAP_SEED,
        "risk": "-log S(36 m) for fitted models; log-odds of death at 2 years for Hosny zero-shot",
        "p_values": "two-sided Wald z with bootstrap standard error; Holm within each family",
        "conditions": {key: {"uno_c_36m": point[key],
                             "ci95": [float(np.nanpercentile(boot[key], 2.5)),
                                      float(np.nanpercentile(boot[key], 97.5))]} for key in risks},
        "chance": family({key: wald(point[key] - 0.5, boot[key]) for key in CONDITIONS}),
        "vs_colipri": family({f"{key} - {REFERENCE[key]}": contrast(key, REFERENCE[key]) for key in CONDITIONS}),
        "baselines": family({f"{key} - baseline/{name}": contrast(key, f"baseline/{name}")
                             for key in CONDITIONS for name in BASELINES}),
        "hosny_rungs": family({f"{a} - {b}": contrast(a, b) for a, b in (
            ("finetune/hosny", "zeroshot/hosny"), ("finetune/hosny", "linear/hosny"),
            ("mlp/hosny", "linear/hosny"), ("mlp/pai", "linear/pai"))}),
    }
    write_json(SELECTED / "statistical_tests_tumour_centred.json", result)
    for key, value in result["conditions"].items():
        print(f"{key:28s} {value['uno_c_36m']:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]")
    for name in ("vs_colipri", "baselines", "hosny_rungs"):
        for key, value in result[name].items():
            print(f"{name:12s} {key:52s} {value['difference']:+.4f} p={value['p']:.4f} holm={value['p_holm']:.4f}")


if __name__ == "__main__":
    main()
