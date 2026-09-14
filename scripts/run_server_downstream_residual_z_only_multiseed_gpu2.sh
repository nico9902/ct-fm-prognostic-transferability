#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"
TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_transitions.csv}"
RUN_DIR="${RUN_DIR:-outputs/latent_predictor_${TAG}_residual_z_only_gpu${GPU_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/downstream_${TAG}_residual_z_only_multiseed_gpu${GPU_ID}}"
OUTCOMES="${OUTCOMES:-cancer_label}"
SEEDS="${SEEDS:-0 1 2 3 4 42 101 202 303 404}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

CHECKPOINTS=("${RUN_DIR}"/best-*.ckpt)
if [[ ! -f "${CHECKPOINTS[0]}" ]]; then
  echo "No checkpoint found in ${RUN_DIR}" >&2
  exit 1
fi
CHECKPOINT="${CHECKPOINTS[0]}"

mkdir -p "${OUTPUT_ROOT}"

for SEED in ${SEEDS}; do
  echo "Running residual downstream seed ${SEED}"
  "${PYTHON}" scripts/run_downstream_clinical_eval.py \
    --checkpoint "${CHECKPOINT}" \
    --transitions "${TRANSITIONS}" \
    --output-dir "${OUTPUT_ROOT}/seed_${SEED}" \
    --outcomes ${OUTCOMES} \
    --no-use-clinical \
    --no-use-delta \
    --prediction-mode residual \
    --seed "${SEED}" \
    --batch-size 32 \
    --num-workers 4
done

"${PYTHON}" scripts/summarize_downstream_multiseed.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
