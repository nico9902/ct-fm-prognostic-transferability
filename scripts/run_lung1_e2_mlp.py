#!/usr/bin/env python3
"""E2 frozen-global MLP Cox rung on the locked LUNG1 repeated nested folds.

One invocation handles one encoder. Outputs are resumable at outer-fold boundaries.
The W&B run uses offline mode by default; set WANDB_MODE=online on an authenticated host.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_lung1_frozen_cox_e1 import (  # noqa: E402
    GRID, INNER_FOLDS, OUTER_FOLDS, REPETITIONS, cohort, fold_metrics,
    inner_assignments, load_embeddings, locked_folds, sha256, uno_c,
)

MODELS = ("tangerine", "radjepa", "colipri", "ctclip")
CONFIGS = ({"hidden": 32, "dropout": 0.1, "weight_decay": 1e-4},
           {"hidden": 64, "dropout": 0.2, "weight_decay": 1e-3})


class CoxMLP(torch.nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dimension, hidden),
                                       torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()


class EfronLoss:
    """Precompute tie groups once so every training epoch is vectorized."""

    def __init__(self, time: np.ndarray, event: np.ndarray):
        order = np.argsort(-time, kind="mergesort")
        sorted_time = time[order]
        sorted_event = event[order]
        ends = np.flatnonzero(np.r_[sorted_time[:-1] != sorted_time[1:], True])
        starts = np.r_[0, ends[:-1] + 1]
        group = np.repeat(np.arange(len(ends)), ends - starts + 1)
        deaths = np.bincount(group, weights=sorted_event, minlength=len(ends)).astype(int)
        active = np.flatnonzero(deaths)
        self.order = torch.as_tensor(order, dtype=torch.long)
        self.group = torch.as_tensor(group, dtype=torch.long)
        self.event = torch.as_tensor(sorted_event.astype(bool))
        self.step_ends = torch.as_tensor(np.repeat(ends[active], deaths[active]), dtype=torch.long)
        self.step_group = torch.as_tensor(np.repeat(active, deaths[active]), dtype=torch.long)
        self.fractions = torch.as_tensor(
            np.concatenate([np.arange(count) / count for count in deaths[active]]),
            dtype=torch.float32)
        self.group_count = len(ends)
        self.death_count = int(sorted_event.sum())

    def __call__(self, risk: torch.Tensor) -> torch.Tensor:
        order = self.order.to(risk.device)
        group = self.group.to(risk.device)
        event = self.event.to(risk.device)
        step_ends = self.step_ends.to(risk.device)
        step_group = self.step_group.to(risk.device)
        fractions = self.fractions.to(risk.device)
        risk = risk[order]
        shift = risk.detach().max()
        weights = torch.exp(risk - shift)
        cumulative = torch.cumsum(weights, 0)
        event_weights = torch.zeros(self.group_count, dtype=risk.dtype, device=risk.device)
        event_weights.scatter_add_(0, group[event], weights[event])
        denominators = cumulative[step_ends] - fractions * event_weights[step_group]
        return (torch.log(denominators).sum() + self.death_count * shift
                - risk[event].sum()) / self.death_count


def fit(x: np.ndarray, time: np.ndarray, event: np.ndarray, config: dict,
        seed: int, epochs: int) -> tuple[CoxMLP, float]:
    torch.manual_seed(seed)
    model = CoxMLP(x.shape[1], config["hidden"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=config["weight_decay"])
    features = torch.as_tensor(x, dtype=torch.float32)
    objective = EfronLoss(time, event)
    model.train()
    loss_value = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(features))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Efron Cox loss")
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return model.eval(), loss_value


def predict(model: CoxMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy().astype(float)


def breslow_survival(train_time: np.ndarray, train_event: np.ndarray,
                     train_risk: np.ndarray, test_risk: np.ndarray) -> np.ndarray:
    """Training-only Breslow baseline after Efron coefficient optimization."""
    shift = float(train_risk.max())
    event_times = np.unique(train_time[train_event == 1])
    increments = np.array([((train_time == t) & (train_event == 1)).sum()
                           / np.exp(np.clip(train_risk[train_time >= t] - shift, -700, 700)).sum()
                           for t in event_times])
    cumulative = np.cumsum(increments)
    positions = np.searchsorted(event_times, GRID, side="right") - 1
    baseline = np.where(positions < 0, 0.0, cumulative[np.maximum(positions, 0)])
    return np.exp(-np.exp(np.clip(test_risk - shift, -700, 700))[:, None] * baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-folds", type=int, default=0, help="Smoke run only; use a separate output root")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/lung1_e2_mlp")
    parser.add_argument("--wandb-project", default="lung1-prognostic-transferability")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_folds < 0:
        parser.error("epochs must be positive and max-folds nonnegative")
    torch.set_num_threads(2)
    data = cohort(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv")
    folds = locked_folds(data, ROOT / "configs/lung1_e1_folds.csv")
    inners = inner_assignments(data, folds)
    manifest = ROOT / f"outputs/lung1_e1_embeddings/{args.model}/original_ct/embedding_manifest.csv"
    name, features, manifest_hash, matrix_hash = load_embeddings(f"{args.model}={manifest}", data)
    output = args.output_root / name
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"method": "frozen_global_mlp_efron_cox", "model": name,
                "cohort_sha256": sha256(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"),
                "folds_sha256": sha256(ROOT / "configs/lung1_e1_folds.csv"),
                "manifest_sha256": manifest_hash, "matrix_sha256": matrix_hash,
                "epochs": args.epochs, "configs": list(CONFIGS), "max_folds": args.max_folds,
                "patients": len(data), "events": int(data.os_event.sum())}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f"Output protocol differs: {protocol_path}")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    metrics_path = output / "fold_metrics.csv"
    predictions_path = output / "out_of_fold_predictions.csv"
    metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    predictions = pd.read_csv(predictions_path).to_dict("records") if predictions_path.exists() else []
    if bool(metrics) != bool(predictions):
        raise ValueError("Metrics and predictions must both exist for resumption")
    completed = {(int(r["repetition"]), int(r["fold"])) for r in metrics}
    if len(completed) != len(metrics):
        raise ValueError("Duplicate outer fold in saved results")
    import wandb
    import os
    run = wandb.init(project=args.wandb_project, name=f"E2-MLP-{name}",
                     group="lung1-e2-original-ct", job_type="mlp-cox",
                     mode=os.environ.get("WANDB_MODE", "offline"),
                     dir=str(output), config=protocol, resume="allow")
    started = time.monotonic()
    try:
        for repetition in range(REPETITIONS):
            for fold in range(OUTER_FOLDS):
                if (repetition, fold) in completed:
                    continue
                if args.max_folds and len(metrics) >= args.max_folds:
                    break
                test_ids = set(folds.loc[(folds.repetition == repetition) &
                                         (folds.fold == fold), "patient_id"])
                test = data.patient_id.isin(test_ids).to_numpy()
                train = ~test
                train_data = data.loc[train].reset_index(drop=True)
                train_x, test_x = features[train], features[test]
                train_time = train_data.os_time_days.to_numpy(dtype=float)
                train_event = train_data.os_event.to_numpy(dtype=int)
                test_time = data.os_time_days.to_numpy(dtype=float)[test]
                test_event = data.os_event.to_numpy(dtype=int)[test]
                inner = inners.loc[(inners.repetition == repetition) & (inners.fold == fold)]
                scores = []
                for ci, config in enumerate(CONFIGS):
                    inner_scores = []
                    for inner_fold in range(INNER_FOLDS):
                        validation = train_data.patient_id.isin(
                            inner.loc[inner.inner_fold == inner_fold, "patient_id"]).to_numpy()
                        scaler = StandardScaler().fit(train_x[~validation])
                        fit_x = scaler.transform(train_x[~validation]).astype(np.float32)
                        val_x = scaler.transform(train_x[validation]).astype(np.float32)
                        seed = 20261005 + repetition * 1000 + fold * 100 + ci * 10 + inner_fold
                        net, _ = fit(fit_x, train_time[~validation], train_event[~validation],
                                     config, seed, args.epochs)
                        inner_scores.append(uno_c(train_time[~validation], train_event[~validation],
                                                  train_time[validation], train_event[validation],
                                                  predict(net, val_x)))
                    if not np.isfinite(inner_scores).all():
                        raise ValueError("Undefined inner-fold Uno C")
                    scores.append(float(np.mean(inner_scores)))
                best = int(np.argmax(scores))
                scaler = StandardScaler().fit(train_x)
                fit_x = scaler.transform(train_x).astype(np.float32)
                eval_x = scaler.transform(test_x).astype(np.float32)
                net, train_loss = fit(fit_x, train_time, train_event, CONFIGS[best],
                                      20261005 + repetition * 1000 + fold * 100 + best * 10 + 9,
                                      args.epochs)
                train_risk = predict(net, fit_x)
                risk = predict(net, eval_x)
                survival = breslow_survival(train_time, train_event, train_risk, risk)
                score = fold_metrics(train_time, train_event, test_time, test_event, risk, survival)
                row = {"model": name, "repetition": repetition, "fold": fold,
                       "config_index": best, "inner_uno_c": scores[best],
                       "train_loss": train_loss, "n_train": int(train.sum()),
                       "n_test": int(test.sum()), "events_train": int(train_event.sum()),
                       "events_test": int(test_event.sum()),
                       "trainable_parameters": sum(p.numel() for p in net.parameters()),
                       "elapsed_seconds": time.monotonic() - started, **score}
                metrics.append(row)
                for i, patient_id in enumerate(data.patient_id[test]):
                    pred = {"model": name, "repetition": repetition, "fold": fold,
                            "patient_id": patient_id, "os_time_days": test_time[i],
                            "os_event": test_event[i], "risk": risk[i]}
                    pred.update({f"survival_{j:02d}": survival[i, j] for j in range(len(GRID))})
                    predictions.append(pred)
                pd.DataFrame(predictions).to_csv(predictions_path, index=False)
                pd.DataFrame(metrics).to_csv(metrics_path, index=False)
                wandb.log({"repetition": repetition, "fold": fold, **{k: float(v) for k, v in score.items()},
                           "inner_uno_c": scores[best], "train_loss": train_loss,
                           "trainable_parameters": row["trainable_parameters"],
                           "elapsed_seconds": row["elapsed_seconds"]})
                print(f"{name} repeat={repetition} fold={fold} Uno C={score['uno_c_36m']:.3f}", flush=True)
        summary = {key: float(pd.DataFrame(metrics)[key].mean()) for key in
                   ("uno_c_36m", "auc_1y", "auc_2y", "auc_3y", "ibs_3m_36m",
                    "calibration_in_large_2y", "calibration_slope")}
        summary.update({"completed_folds": len(metrics), "expected_folds": REPETITIONS * OUTER_FOLDS})
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        wandb.summary.update(summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
