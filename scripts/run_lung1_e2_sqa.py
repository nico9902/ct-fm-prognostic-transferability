#!/usr/bin/env python3
"""E2 Survival Query Adapter with a frozen CT encoder and cached spatial tokens.

Uses the E1 patient folds and a nested shared two-point regularization grid.
Run on a CUDA host after extract_lung1_e2_tokens.py completes for each model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_lung1_e2_mlp import EfronLoss, breslow_survival  # noqa: E402
from run_lung1_frozen_cox_e1 import (  # noqa: E402
    GRID, INNER_FOLDS, OUTER_FOLDS, REPETITIONS, cohort, fold_metrics,
    inner_assignments, locked_folds, sha256, uno_c,
)

MODELS = ("tangerine", "radjepa", "colipri", "ctclip")
CONFIGS = ({"dropout": 0.1, "weight_decay": 1e-4},
           {"dropout": 0.2, "weight_decay": 1e-3})


def spatial_positions(grid: tuple[int, int, int], dim: int = 128) -> torch.Tensor:
    axes = [torch.linspace(-1, 1, steps=n) for n in grid]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    frequencies = 2 ** torch.arange(20, dtype=torch.float32) / 16
    phase = coordinates[:, :, None] * frequencies[None, None, :] * torch.pi
    encoding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=2).flatten(1)
    return torch.nn.functional.pad(encoding, (0, dim - encoding.shape[1])).unsqueeze(0)


class SurvivalQueryAdapter(torch.nn.Module):
    def __init__(self, input_dim: int, dropout: float,
                 grid: tuple[int, int, int] | None = None):
        super().__init__()
        dim = 128
        self.projection = torch.nn.Linear(input_dim, dim)
        self.norm = torch.nn.LayerNorm(dim)
        self.register_buffer("position", spatial_positions(grid, dim) if grid else None)
        self.queries = torch.nn.Parameter(torch.randn(4, dim) * 0.02)
        self.attention = torch.nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.block = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, dim),
                                         torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.head = torch.nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.projection(tokens))
        if self.position is not None:
            x = x + self.position
        query = self.queries.unsqueeze(0).expand(len(tokens), -1, -1)
        attended, _ = self.attention(query, x, x, need_weights=False)
        features = (attended + self.block(attended)).mean(dim=1)
        return self.head(features).flatten()


def load_tokens(model: str, data: pd.DataFrame) -> tuple[np.ndarray, dict]:
    folder = ROOT / f"outputs/lung1_e2_tokens/{model}/original_ct"
    path = folder / "token_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Extract all final tokens first: {path}")
    manifest = pd.read_csv(path, dtype={"patient_id": str, "ct_series_uid": str})
    if manifest.patient_id.duplicated().any() or set(manifest.patient_id) != set(data.patient_id):
        raise ValueError("Token cache must contain the exact locked cohort")
    manifest = manifest.set_index("patient_id").loc[data.patient_id]
    if not np.array_equal(manifest.ct_series_uid.to_numpy(), data.ct_series_uid.to_numpy()):
        raise ValueError("Token series UIDs differ from the locked cohort")
    arrays = []
    shape = None
    for _, row in manifest.iterrows():
        array = np.load(folder / row.token_path, allow_pickle=False)
        if array.ndim != 2 or not np.isfinite(array).all() or array.shape[0] > 512:
            raise ValueError("Tokens must be finite 2-D arrays with at most 512 tokens")
        if shape is None:
            shape = array.shape
        if array.shape != shape:
            raise ValueError("Token shape changes within encoder cohort")
        arrays.append(array)
    grids = {str(value) for value in manifest.saved_grid}
    if len(grids) != 1:
        raise ValueError("Saved token grids must match within an encoder")
    grid = tuple(json.loads(next(iter(grids))))
    if int(np.prod(grid)) != int(shape[0]):
        raise ValueError("Saved token grid does not match token arrays")
    provenance = {"manifest_sha256": sha256(path),
                  "token_protocol_sha256": sha256(folder / "protocol.json"),
                  "shape": list(shape), "saved_grid": list(grid)}
    return np.stack(arrays), provenance


def normalize_tokens(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Equalizes channel scales using only patients in the current training split.
    mean = train.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    std = train.astype(np.float32).std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((train.astype(np.float32) - mean) / std,
            (other.astype(np.float32) - mean) / std)


def predict(net: SurvivalQueryAdapter, x: np.ndarray, device: torch.device,
            batch_size: int = 16) -> np.ndarray:
    net.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            chunks.append(net(torch.as_tensor(x[start:start + batch_size], device=device)).cpu().numpy())
    return np.concatenate(chunks).astype(float)


def fit(x: np.ndarray, time_: np.ndarray, event: np.ndarray, config: dict,
        epochs: int, seed: int, device: torch.device, batch_size: int,
        grid: tuple[int, int, int] | None = None):
    torch.manual_seed(seed)
    net = SurvivalQueryAdapter(x.shape[-1], config["dropout"], grid).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4,
                                  weight_decay=config["weight_decay"])
    criterion = EfronLoss(time_, event)
    # Full risk set is exact; patient microbatches only reduce forward activation peaks.
    net.train()
    loss_value = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        risk = torch.cat([net(torch.as_tensor(x[start:start + batch_size], device=device))
                          for start in range(0, len(x), batch_size)])
        loss = criterion(risk)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite SQA Cox loss")
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return net.eval(), loss_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/lung1_e2_sqa")
    parser.add_argument("--wandb-project", default="lung1-prognostic-transferability")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_folds < 0:
        parser.error("epochs and batch-size must be positive; max-folds must be nonnegative")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("E2 SQA requires a visible CUDA GPU")
    device = torch.device(args.device)
    data = cohort(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv")
    folds = locked_folds(data, ROOT / "configs/lung1_e1_folds.csv")
    inners = inner_assignments(data, folds)
    tokens, token_provenance = load_tokens(args.model, data)
    output = args.output_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"method": "frozen_token_sqa_efron_cox", "model": args.model,
                "cohort_sha256": sha256(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"),
                "folds_sha256": sha256(ROOT / "configs/lung1_e1_folds.csv"),
                "tokens": token_provenance, "adapter_dim": 128, "queries": 4,
                "attention_blocks": 1, "epochs": args.epochs, "batch_size": args.batch_size,
                "configs": list(CONFIGS), "max_folds": args.max_folds}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a new output root")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    metric_path = output / "fold_metrics.csv"
    prediction_path = output / "out_of_fold_predictions.csv"
    metrics = pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else []
    predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    if bool(metrics) != bool(predictions):
        raise ValueError("Metrics and predictions must both exist for resumption")
    completed = {(int(r["repetition"]), int(r["fold"])) for r in metrics}
    if len(completed) != len(metrics):
        raise ValueError("Duplicate completed outer fold")
    import wandb
    run_id = hashlib.sha1(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:8]
    run = wandb.init(project=args.wandb_project, name=f"E2-SQA-{args.model}",
                     group="lung1-e2-original-ct", job_type="sqa-cox",
                     mode=os.environ.get("WANDB_MODE", "offline"),
                     dir=str(output), config=protocol, id=run_id, resume="allow")
    start_time = time.monotonic()
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
                        fit_x, val_x = normalize_tokens(tokens[train][~validation],
                                                         tokens[train][validation])
                        seed = 20261005 + repetition * 1000 + fold * 100 + ci * 10 + inner_fold
                        net, _ = fit(fit_x, train_time[~validation], train_event[~validation],
                                     config, args.epochs, seed, device, args.batch_size,
                                     tuple(token_provenance["saved_grid"]))
                        inner_scores.append(uno_c(train_time[~validation], train_event[~validation],
                                                  train_time[validation], train_event[validation],
                                                  predict(net, val_x, device)))
                        del net
                    if not np.isfinite(inner_scores).all():
                        raise ValueError("Undefined inner Uno C")
                    scores.append(float(np.mean(inner_scores)))
                best = int(np.argmax(scores))
                fit_x, eval_x = normalize_tokens(tokens[train], tokens[test])
                net, train_loss = fit(fit_x, train_time, train_event, CONFIGS[best],
                                      args.epochs, 20261005 + repetition * 1000 + fold * 100 + best * 10 + 9,
                                      device, args.batch_size, tuple(token_provenance["saved_grid"]))
                train_risk = predict(net, fit_x, device)
                risk = predict(net, eval_x, device)
                survival = breslow_survival(train_time, train_event, train_risk, risk)
                score = fold_metrics(train_time, train_event, test_time, test_event, risk, survival)
                row = {"model": args.model, "repetition": repetition, "fold": fold,
                       "config_index": best, "inner_uno_c": scores[best],
                       "train_loss": train_loss, "trainable_parameters": sum(p.numel() for p in net.parameters()),
                       "elapsed_seconds": time.monotonic() - start_time, **score}
                if device.type == "cuda":
                    row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                metrics.append(row)
                for i, patient_id in enumerate(data.patient_id[test]):
                    pred = {"model": args.model, "repetition": repetition, "fold": fold,
                            "patient_id": patient_id, "os_time_days": test_time[i],
                            "os_event": test_event[i], "risk": risk[i]}
                    pred.update({f"survival_{j:02d}": survival[i, j] for j in range(len(GRID))})
                    predictions.append(pred)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
                pd.DataFrame(metrics).to_csv(metric_path, index=False)
                wandb.log({"repetition": repetition, "fold": fold,
                           **{k: float(v) for k, v in score.items()},
                           "inner_uno_c": scores[best], "train_loss": train_loss,
                           "trainable_parameters": row["trainable_parameters"],
                           "peak_gpu_memory_bytes": row.get("peak_gpu_memory_bytes", 0),
                           "elapsed_seconds": row["elapsed_seconds"]})
                print(f"{args.model} repeat={repetition} fold={fold} Uno C={score['uno_c_36m']:.3f}", flush=True)
                del net
        frame = pd.DataFrame(metrics)
        summary = {key: float(frame[key].mean()) for key in
                   ("uno_c_36m", "auc_1y", "auc_2y", "auc_3y", "ibs_3m_36m",
                    "calibration_in_large_2y", "calibration_slope")}
        summary.update({"completed_folds": len(metrics), "expected_folds": REPETITIONS * OUTER_FOLDS})
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        wandb.summary.update(summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
