#!/usr/bin/env bash
# One locked five-fold repetition; all rungs serially on a BF16-capable A100.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); assert "A100" in torch.cuda.get_device_name(0)'; then
  echo 'A visible BF16-capable A100 is required.' >&2
  exit 2
fi
mkdir -p logs/lung1_bf16_selected outputs/lung1_bf16_selected
export MPLCONFIGDIR="$PWD/logs/lung1_bf16_selected/matplotlib"
export WANDB_CACHE_DIR="$PWD/logs/lung1_bf16_selected/wandb_cache"
export WANDB_MODE="${WANDB_MODE:-offline}"
for method in linear mlp sqa lora partial; do
  for model in tangerine radjepa colipri ctclip; do
    echo "Starting ${method}/${model}" >&2
    if [[ "$method" == lora || "$method" == partial ]]; then
      .venv/bin/python -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
        --method "$method" --model "$model" --wandb-mode "$WANDB_MODE" \
        > "logs/lung1_bf16_selected/${method}_${model}.log" 2>&1
    else
      .venv/bin/python -u scripts/run_lung1_bf16_selected_heads.py \
        --method "$method" --model "$model" --wandb-mode "$WANDB_MODE" \
        > "logs/lung1_bf16_selected/${method}_${model}.log" 2>&1
    fi
  done
done
printf 'complete\n' > outputs/lung1_bf16_selected/COMPLETE
