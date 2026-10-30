#!/usr/bin/env bash
# E4 token masking extraction: all five frozen encoders in parallel on one GPU.
# Usage: CUDA_VISIBLE_DEVICES=<A100 UUID> bash scripts/launch_lung1_e4_token_masking.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_e4_token_masking
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
pids=()
for model in colipri tangerine radjepa ctclip dinov3; do
  python=.venv/bin/python
  [[ $model == dinov3 ]] && python=.venv-dinov3/bin/python
  (cd scripts && ../$python extract_lung1_e4_token_masked_embeddings.py --model "$model" --device cuda:0) \
    > "logs/lung1_e4_token_masking/extract_${model}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
