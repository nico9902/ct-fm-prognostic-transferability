#!/usr/bin/env bash
# DINOv3 ViT-B/16 on the locked one-repetition five-fold split: frozen
# extraction, then all five rungs serially on a BF16-capable A100.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-dinov3/bin/python  # transformers >= 4.56 layered on .venv
if ! "$PY" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); assert "A100" in torch.cuda.get_device_name(0)'; then
  echo 'A visible BF16-capable A100 is required.' >&2
  exit 2
fi
LOGS=logs/lung1_bf16_selected
mkdir -p "$LOGS" outputs/lung1_bf16_selected
export MPLCONFIGDIR="$PWD/$LOGS/matplotlib"
export WANDB_CACHE_DIR="$PWD/$LOGS/wandb_cache"
export WANDB_MODE="${WANDB_MODE:-offline}"
echo 'Extracting DINOv3 embeddings, tokens and inputs' >&2
"$PY" -u scripts/extract_lung1_e1_embeddings.py --model dinov3 > "$LOGS/extract_embeddings_dinov3.log" 2>&1
"$PY" -u scripts/extract_lung1_e2_tokens.py --model dinov3 > "$LOGS/extract_tokens_dinov3.log" 2>&1
"$PY" -u scripts/cache_lung1_e2_inputs.py --model dinov3 > "$LOGS/cache_inputs_dinov3.log" 2>&1
for method in linear mlp sqa lora partial; do
  echo "Starting ${method}/dinov3" >&2
  if [[ "$method" == lora || "$method" == partial ]]; then
    "$PY" -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
      --method "$method" --model dinov3 --wandb-mode "$WANDB_MODE" \
      >> "$LOGS/${method}_dinov3.log" 2>&1
  else
    "$PY" -u scripts/run_lung1_bf16_selected_heads.py \
      --method "$method" --model dinov3 --wandb-mode "$WANDB_MODE" \
      >> "$LOGS/${method}_dinov3.log" 2>&1
  fi
done
printf 'complete\n' > outputs/lung1_bf16_selected/COMPLETE_dinov3
