#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
RUN_OUTPUT="${RUN_OUTPUT:-outputs/lung1_e5_lora}"
RUN_LOGS="${RUN_LOGS:-logs/lung1_e5_lora}"
RUN_PRECISION="${RUN_PRECISION:-bf16-mixed}"
export MPLCONFIGDIR="$PWD/$RUN_LOGS/matplotlib"
mkdir -p "$RUN_LOGS" "$RUN_OUTPUT"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$RUN_OUTPUT/FAILED"; fi' EXIT
rm -f "$RUN_OUTPUT/FAILED" "$RUN_OUTPUT/COMPLETE"
for model in colipri tangerine; do
  .venv/bin/python -u scripts/run_lung1_e5_lora.py --precision "$RUN_PRECISION" --output-root "$RUN_OUTPUT" --model "$model" --stage train \
    > "$RUN_LOGS/train_$model.log" 2>&1
done
for model in colipri tangerine; do
  .venv/bin/python -u scripts/cache_lung1_e2_inputs.py --model "$model" \
    --cohort data/derived/radiogenomics/cohort_with_gtv_volume.csv \
    --status-column global_status --clamp-min-hu -1024 \
    --output-root outputs/radiogenomics_lora_inputs \
    > "$RUN_LOGS/cache_$model.log" 2>&1
  .venv/bin/python -u scripts/run_lung1_e5_lora.py --precision "$RUN_PRECISION" --output-root "$RUN_OUTPUT" --model "$model" --stage predict \
    > "$RUN_LOGS/predict_$model.log" 2>&1
done
for model in colipri tangerine; do
  .venv/bin/python -u scripts/run_lung1_e5_lora.py --precision "$RUN_PRECISION" --output-root "$RUN_OUTPUT" --model "$model" --stage evaluate \
    > "$RUN_LOGS/evaluate_$model.log" 2>&1
done
date -Is > "$RUN_OUTPUT/COMPLETE"
