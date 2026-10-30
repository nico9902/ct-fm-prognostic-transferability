#!/usr/bin/env bash
# Five locked outer folds, one repetition, serial LoRA then partial BF16 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); assert "A100" in torch.cuda.get_device_name(0)'; then
  echo 'A visible BF16-capable A100 is required.' >&2
  exit 2
fi
mkdir -p logs/lung1_e2_bf16 outputs/lung1_e2_encoder_bf16_lightning
export MPLCONFIGDIR="$PWD/logs/lung1_e2_bf16/matplotlib"
export WANDB_CACHE_DIR="$PWD/logs/lung1_e2_bf16/wandb_cache"
export WANDB_MODE="${WANDB_MODE:-offline}"
for method in lora partial; do
  for model in tangerine radjepa colipri ctclip; do
    .venv/bin/python -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
      --method "$method" --model "$model" --wandb-mode "$WANDB_MODE" \
      > "logs/lung1_e2_bf16/${method}_${model}.log" 2>&1
  done
done
printf 'complete\n' > outputs/lung1_e2_encoder_bf16_lightning/COMPLETE
