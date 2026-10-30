#!/usr/bin/env bash
# Nested-CV inner stage for LoRA/partial (docs/lung1_sweep_validation_protocol.md).
# Usage: launch_lung1_nested_inner.sh K   (K = 1 or 2; inner fold 0 is launch_lung1_sweep_valsel.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
k="${1:?inner fold (1 or 2) required}"
[[ "$k" == 1 || "$k" == 2 ]] || { echo "inner fold must be 1 or 2" >&2; exit 2; }
out="outputs/lung1_nested/inner${k}"; logs="logs/lung1_nested/inner${k}"
mkdir -p "$logs" "$out"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
export MPLCONFIGDIR="$PWD/$logs/matplotlib"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$out/FAILED"; fi' EXIT
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "A100" in torch.cuda.get_device_name(0), "A100 required"; print(torch.cuda.get_device_name(0), flush=True)'
rm -f "$out/FAILED" "$out/COMPLETE"
for method in lora partial; do
  if [[ "$method" == lora ]]; then rates=(1e-5 1e-4 1e-3); else rates=(1e-6 1e-5 1e-4); fi
  for model in colipri tangerine; do
    for rate in "${rates[@]}"; do
      tag="${method}_${model}_lr${rate}_ep20"
      echo "START $tag $(date -Is)"
      .venv/bin/python -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
        --method "$method" --model "$model" --learning-rate "$rate" \
        --epochs 20 --validation-inner-fold "$k" --wandb-mode offline \
        --output-root "$out/lr${rate}_ep20" > "$logs/${tag}.log" 2>&1
      echo "DONE $tag $(date -Is)"
    done
  done
done
echo "COMPLETE $(date -Is)" > "$out/COMPLETE"
