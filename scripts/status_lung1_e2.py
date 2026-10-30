#!/usr/bin/env python3
"""Print progress of the resumable E2 background queues."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("tangerine", "radjepa", "colipri", "ctclip")


def count(path: Path) -> int:
    return len(pd.read_csv(path)) if path.is_file() else 0


print("model        tokens  inputs  mlp  sqa  lora  partial  (expected: 416 patients / 25 folds)")
for model in MODELS:
    values = (
        count(ROOT / f"outputs/lung1_e2_tokens/{model}/original_ct/token_manifest.csv"),
        count(ROOT / f"outputs/lung1_e2_inputs/{model}/original_ct/input_manifest.csv"),
        count(ROOT / f"outputs/lung1_e2_mlp/{model}/fold_metrics.csv"),
        count(ROOT / f"outputs/lung1_e2_sqa/{model}/fold_metrics.csv"),
        count(ROOT / f"outputs/lung1_e2_encoder_adaptation/lora/{model}/fold_metrics.csv"),
        count(ROOT / f"outputs/lung1_e2_encoder_adaptation/partial/{model}/fold_metrics.csv"),
    )
    print(f"{model:12} {values[0]:6} {values[1]:7} {values[2]:4} {values[3]:4} {values[4]:5} {values[5]:8}")
for name in ("lung1_e2_mlp", "lung1_e2_inputs", "lung1_e2_sqa", "lung1_e2_encoder_adaptation"):
    marker = ROOT / "outputs" / name / "COMPLETE"
    print(f"{name}: {'complete' if marker.is_file() else 'running/pending'}")

print("\nBF16 Lightning encoder adaptation (one 5-fold repetition):")
print("model        lora  partial  (expected: 5 folds each)")
for model in MODELS:
    values = [count(ROOT / f"outputs/lung1_e2_encoder_bf16_lightning/{method}/{model}/fold_metrics.csv")
              for method in ("lora", "partial")]
    print(f"{model:12} {values[0]:4} {values[1]:8}")
marker = ROOT / "outputs/lung1_e2_encoder_bf16_lightning/COMPLETE"
print(f"lung1_e2_encoder_bf16_lightning: {'complete' if marker.is_file() else 'running/pending'}")

print("\nBF16 selected-checkpoint protocol (one 5-fold repetition):")
print("model        linear  mlp  sqa  lora  partial  (expected: 5 folds each)")
for model in MODELS:
    values = [count(ROOT / f"outputs/lung1_bf16_selected/{method}/{model}/fold_metrics.csv")
              for method in ("linear", "mlp", "sqa", "lora", "partial")]
    print(f"{model:12} {values[0]:6} {values[1]:4} {values[2]:4} {values[3]:5} {values[4]:8}")
marker = ROOT / "outputs/lung1_bf16_selected/COMPLETE"
print(f"lung1_bf16_selected: {'complete' if marker.is_file() else 'running/pending'}")
