#!/usr/bin/env bash
# Execute the frozen-token SQA rung after moving this checkout to a CUDA host.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available()'; then
  echo 'A visible CUDA GPU is required for LUNG1 E2 token extraction and SQA.' >&2
  exit 2
fi
mkdir -p logs/lung1_e2/wandb_cache outputs/lung1_e2_sqa
export WANDB_CACHE_DIR="$PWD/logs/lung1_e2/wandb_cache"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-lung1-prognostic-transferability}"

for model in tangerine radjepa colipri ctclip; do
  .venv/bin/python scripts/extract_lung1_e2_tokens.py --model "$model" \
    > "logs/lung1_e2/tokens_${model}.log" 2>&1
  .venv/bin/python scripts/run_lung1_e2_sqa.py --model "$model" \
    --wandb-project "$WANDB_PROJECT" > "logs/lung1_e2/sqa_${model}.log" 2>&1
done
.venv/bin/python scripts/compare_lung1_e2.py > logs/lung1_e2/paired_comparison_with_sqa.log 2>&1
printf 'complete\n' > outputs/lung1_e2_sqa/COMPLETE
