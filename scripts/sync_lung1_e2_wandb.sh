#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_e2/wandb_cache
export WANDB_CACHE_DIR="$PWD/logs/lung1_e2/wandb_cache"
while IFS= read -r run; do
  .venv/bin/wandb sync "$run"
done < <(find outputs/lung1_e1_frozen_cox_four_models/wandb \
               outputs/lung1_e2_mlp outputs/lung1_e2_comparison \
               -type d -name 'offline-run-*' 2>/dev/null | sort)
