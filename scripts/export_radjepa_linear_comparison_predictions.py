#!/usr/bin/env python3
"""Reproduce historical best pure-linear Rad-JEPA probes for keyed sensitivity predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

import run_diagnostic_state_decodability as study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/tangerine_decodability/radjepa_reproduced"))
    p.add_argument("--gpu", type=int, default=1)
    a = p.parse_args()
    torch.cuda.set_device(a.gpu)
    original_trainer = study.pl.Trainer
    def trainer_on_requested_gpu(**kwargs):
        kwargs["devices"] = [a.gpu]
        return original_trainer(**kwargs)
    study.pl.Trainer = trainer_on_requested_gpu
    torch.set_num_threads(4)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.root / "experiment_scan_manifest.csv", dtype=str)
    baseline = pd.read_csv(a.root / "summary_results.csv")
    baseline = baseline[baseline.representation.ne("sliding_gatedattention")]
    wins = np.load(a.root / "cache/sliding_window_embeddings.npy", mmap_mode="r")
    wm = pd.read_csv(a.root / "cache/sliding_window_metadata.csv")
    by_scan = {int(k): v.to_numpy() for k, v in wm.groupby("scan_index").groups.items()}
    mean = np.vstack([wins[by_scan[i]].mean(0) for i in range(len(d))])
    maximum = np.vstack([wins[by_scan[i]].max(0) for i in range(len(d))])
    reps = {"resize32": np.load(a.root / "cache/resize32_embeddings.npy"), "sliding_mean": mean,
            "sliding_max": maximum, "sliding_meanmax": np.hstack([mean, maximum])}
    tr, va, te = [d.split.eq(s).to_numpy() for s in ["train", "val", "test"]]
    original_metrics = study.metrics
    captured = {}
    def capture(y, prob, threshold=.5):
        captured["prob"] = prob.copy()
        return original_metrics(y, prob, threshold)
    study.metrics = capture
    rows = []
    for task in study.TASKS:
        rep = baseline[baseline.task.eq(task)].sort_values("auroc_mean", ascending=False).iloc[0].representation
        x = reps[rep]
        x = StandardScaler().fit(x[tr]).transform(x).astype(np.float32)
        y = pd.to_numeric(d[task]).to_numpy(dtype=np.float32)
        for seed in range(5):
            result, _ = study.train_one(x[tr], y[tr], x[va], y[va], x[te], y[te], rep, seed, 128, 100, 15)
            pred = d.loc[te, ["patient_id", "timepoint", "series_uid"]].copy()
            pred["label"] = y[te]
            pred["probability"] = captured["prob"]
            pred.to_csv(a.output_dir / f"{task}_{seed}_test_predictions.csv", index=False)
            rows.append({"task": task, "representation": rep, "seed": seed, **result})
            pd.DataFrame(rows).to_csv(a.output_dir / "per_seed_results.csv", index=False)
            print(task, rep, seed, result, flush=True)


if __name__ == "__main__":
    main()
