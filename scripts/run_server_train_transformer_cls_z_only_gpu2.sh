#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"

TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_transitions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/latent_predictor_${TAG}_transformer_cls_z_only_gpu${GPU_ID}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_latent_predictor_${TAG}_transformer_cls_z_only_gpu${GPU_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/train_latent_predictor_${TAG}_transformer_cls_z_only_gpu${GPU_ID}.pid}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-32-true}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
TRANSFORMER_DIM="${TRANSFORMER_DIM:-384}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-8}"
TRANSFORMER_MLP_DIM="${TRANSFORMER_MLP_DIM:-1536}"
WANDB_PROJECT="${WANDB_PROJECT:-lung-jepa-world-model}"
WANDB_NAME="${WANDB_NAME:-radjepa3d_${TAG}_transformer_cls_z_only_gpu${GPU_ID}}"
WANDB_MODE_FLAG="${WANDB_MODE_FLAG:---wandb}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

if [[ ! -f "${TRANSITIONS}" ]]; then
  echo "Missing transitions file: ${TRANSITIONS}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${PYTHON}" -m lung_jepa.training.train_latent_predictor \
  --transitions "${TRANSITIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --val-fraction "${VAL_FRACTION}" \
  --test-fraction "${TEST_FRACTION}" \
  --no-use-clinical \
  --no-use-delta \
  --model-arch transformer_cls \
  --prediction-mode direct \
  --transformer-dim "${TRANSFORMER_DIM}" \
  --transformer-layers "${TRANSFORMER_LAYERS}" \
  --transformer-heads "${TRANSFORMER_HEADS}" \
  --transformer-mlp-dim "${TRANSFORMER_MLP_DIM}" \
  --test \
  --accelerator gpu \
  --devices 1 \
  --precision "${PRECISION}" \
  "${WANDB_MODE_FLAG}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-name "${WANDB_NAME}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"
echo "Started transformer_cls z_only training PID ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
