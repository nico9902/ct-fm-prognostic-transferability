#!/usr/bin/env bash
# Nested-CV inner stage for the MLP and SQA readouts, inner folds 0-2
# (docs/lung1_sweep_validation_protocol.md).
set -euo pipefail
cd "$(dirname "$0")/.."
out="outputs/lung1_nested/heads"; logs="logs/lung1_nested/heads"
mkdir -p "$logs" "$out"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
export MPLCONFIGDIR="$PWD/$logs/matplotlib"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$out/FAILED_inner"; fi' EXIT
rm -f "$out/FAILED_inner" "$out/COMPLETE_inner"
for k in 0 1 2; do
  for method in mlp sqa; do
    if [[ "$method" == mlp ]]; then models=(colipri tangerine radjepa ctclip dinov3 pai hosny)
    else models=(colipri tangerine radjepa ctclip dinov3); fi
    for model in "${models[@]}"; do
      tag="inner${k}_${method}_${model}"
      echo "START $tag $(date -Is)"
      .venv/bin/python -u scripts/run_lung1_bf16_selected_heads.py \
        --method "$method" --model "$model" --validation-inner-fold "$k" \
        --wandb-mode offline --output-root "$out/inner${k}" > "$logs/${tag}.log" 2>&1
      echo "DONE $tag $(date -Is)"
    done
  done
done
echo "COMPLETE $(date -Is)" > "$out/COMPLETE_inner"
