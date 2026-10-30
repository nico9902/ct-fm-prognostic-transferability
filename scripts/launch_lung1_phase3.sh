#!/usr/bin/env bash
# Predeclared sensitivity sweep; each setting gets all five locked outer folds.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_phase3 outputs/lung1_phase3
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
export MPLCONFIGDIR="$PWD/logs/lung1_phase3/matplotlib"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > outputs/lung1_phase3/FAILED; fi' EXIT
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "A100" in torch.cuda.get_device_name(0), "A100 required"; print(torch.cuda.get_device_name(0), flush=True)'
rm -f outputs/lung1_phase3/FAILED outputs/lung1_phase3/COMPLETE
for method in lora partial; do
  if [[ "$method" == lora ]]; then
    rates=(1e-5 1e-4 1e-3)
  else
    rates=(1e-6 1e-5 1e-4)
  fi
  for model in colipri tangerine; do
    for rate in "${rates[@]}"; do
      for epochs in 5 20; do
        tag="${method}_${model}_lr${rate}_ep${epochs}"
        echo "START $tag $(date -Is)"
        .venv/bin/python -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
          --method "$method" --model "$model" --learning-rate "$rate" \
          --epochs "$epochs" --fixed-trainval --wandb-mode offline \
          --output-root "outputs/lung1_phase3/lr${rate}_ep${epochs}" \
          > "logs/lung1_phase3/${tag}.log" 2>&1
        echo "DONE $tag $(date -Is)"
      done
    done
  done
done
rm -f outputs/lung1_phase3/FAILED
date -Is > outputs/lung1_phase3/COMPLETE
