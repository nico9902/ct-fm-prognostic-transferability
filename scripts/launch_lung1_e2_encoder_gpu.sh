#!/usr/bin/env bash
# Secondary E2 image-level adaptation rungs. Run on a GPU after input caches pass QC.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available()'; then
  echo 'A visible CUDA GPU is required for encoder adaptation.' >&2
  exit 2
fi
if [[ ! -f outputs/lung1_e2_inputs/COMPLETE ]]; then
  echo 'Complete the image input cache before encoder adaptation.' >&2
  exit 2
fi
mkdir -p logs/lung1_e2/wandb_cache outputs/lung1_e2_encoder_adaptation
export WANDB_CACHE_DIR="$PWD/logs/lung1_e2/wandb_cache"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-lung1-prognostic-transferability}"
for method in lora partial; do
  for model in tangerine radjepa colipri ctclip; do
    .venv/bin/python scripts/run_lung1_e2_encoder_adaptation.py \
      --model "$model" --method "$method" --wandb-project "$WANDB_PROJECT" \
      > "logs/lung1_e2/${method}_${model}.log" 2>&1
  done
done
.venv/bin/python scripts/compare_lung1_e2.py \
  > logs/lung1_e2/paired_comparison_with_encoder_adaptation.log 2>&1
printf 'complete\n' > outputs/lung1_e2_encoder_adaptation/COMPLETE
