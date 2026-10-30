#!/usr/bin/env python3
"""Register the locked E1 Linear Cox rung as E2 W&B runs without refitting it."""

import json
import os
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/lung1_e1_frozen_cox_four_models"
metrics = pd.read_csv(SOURCE / "fold_metrics.csv")
protocol = json.loads((SOURCE / "protocol.json").read_text())
if len(metrics) != 100 or metrics.groupby("model").size().nunique() != 1:
    raise ValueError("The locked E1 baseline must contain 25 folds for each of four models")
for name, rows in metrics.groupby("model", sort=True):
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "lung1-prognostic-transferability"),
                     name=f"E2-Linear-{name}", group="lung1-e2-original-ct",
                     job_type="linear-cox", mode=os.environ.get("WANDB_MODE", "offline"),
                     dir=str(SOURCE), config={"method": "frozen_global_linear_cox_e1_reuse",
                                             "model": name, "source_protocol": protocol,
                                             "source_dir": str(SOURCE)})
    for _, row in rows.sort_values(["repetition", "fold"]).iterrows():
        wandb.log({key: float(value) for key, value in row.items()
                   if key != "model" and pd.notna(value)})
    run.summary.update({key: float(rows[key].mean()) for key in
                        ("uno_c_36m", "auc_1y", "auc_2y", "auc_3y", "ibs_3m_36m",
                         "calibration_in_large_2y", "calibration_slope")})
    run.summary.update({"completed_folds": len(rows), "source": str(SOURCE)})
    run.finish()
    print(f"Logged E2 linear baseline: {name}", flush=True)
