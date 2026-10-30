#!/usr/bin/env bash
# Run the E2 rungs supported by the existing frozen global embeddings.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_e2/wandb_cache outputs/lung1_e2_mlp
export WANDB_CACHE_DIR="$PWD/logs/lung1_e2/wandb_cache"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-lung1-prognostic-transferability}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

.venv/bin/python scripts/log_lung1_e2_linear_baseline.py > logs/lung1_e2/linear.log 2>&1
for model in tangerine radjepa colipri ctclip; do
  .venv/bin/python scripts/run_lung1_e2_mlp.py --model "$model" \
    --wandb-project "$WANDB_PROJECT" > "logs/lung1_e2/mlp_${model}.log" 2>&1
done
printf 'complete\n' > outputs/lung1_e2_mlp/COMPLETE
