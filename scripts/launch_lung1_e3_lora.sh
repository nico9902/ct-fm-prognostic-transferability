#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
RUN_OUTPUT="${RUN_OUTPUT:-outputs/lung1_e3_lora}"
RUN_LOGS="${RUN_LOGS:-logs/lung1_e3_lora}"
RUN_PRECISION="${RUN_PRECISION:-bf16-mixed}"
export MPLCONFIGDIR="$PWD/$RUN_LOGS/matplotlib"
mkdir -p "$RUN_LOGS" "$RUN_OUTPUT"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$RUN_OUTPUT/FAILED"; fi' EXIT
rm -f "$RUN_OUTPUT/FAILED" "$RUN_OUTPUT/COMPLETE"
for model in colipri tangerine; do
  for budget in 100 50 25 10; do
    draws=10
    if [[ "$budget" == 100 ]]; then draws=1; fi
    for ((draw=0; draw<draws; draw++)); do
      tag="${model}_pct${budget}_draw${draw}"
      echo "START $tag $(date -Is)"
      .venv/bin/python -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
        --precision "$RUN_PRECISION" --method lora --model "$model" --learning-rate 1e-3 --epochs 20 --fixed-trainval \
        --budget-pct "$budget" --draw "$draw" --wandb-mode offline \
        --output-root "$RUN_OUTPUT/pct${budget}/draw${draw}" \
        > "$RUN_LOGS/$tag.log" 2>&1
      echo "DONE $tag $(date -Is)"
    done
  done
done
.venv/bin/python scripts/summarize_lung1_e3_lora.py --output-root "$RUN_OUTPUT"
date -Is > "$RUN_OUTPUT/COMPLETE"
