#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available()'; then
  echo 'A visible CUDA GPU is required to cache CT-CLIP image inputs.' >&2
  exit 2
fi
mkdir -p logs/lung1_e2 outputs/lung1_e2_inputs
for model in tangerine radjepa colipri ctclip; do
  .venv/bin/python scripts/cache_lung1_e2_inputs.py --model "$model" \
    > "logs/lung1_e2/inputs_${model}.log" 2>&1
done
printf 'complete\n' > outputs/lung1_e2_inputs/COMPLETE
