#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-1}"

TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_transitions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/latent_predictor_${TAG}_gpu${GPU_ID}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_latent_predictor_${TAG}_gpu${GPU_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/train_latent_predictor_${TAG}_gpu${GPU_ID}.pid}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-32-true}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
USE_CLINICAL="${USE_CLINICAL:-1}"
USE_DELTA="${USE_DELTA:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-lung-jepa-world-model}"
WANDB_NAME="${WANDB_NAME:-radjepa3d_${TAG}_mlp_gpu${GPU_ID}}"
WANDB_MODE_FLAG="${WANDB_MODE_FLAG:---wandb}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

CLINICAL_FLAG="--use-clinical"
if [[ "${USE_CLINICAL}" == "0" ]]; then
  CLINICAL_FLAG="--no-use-clinical"
fi

DELTA_FLAG="--use-delta"
if [[ "${USE_DELTA}" == "0" ]]; then
  DELTA_FLAG="--no-use-delta"
fi

if [[ ! -f "${TRANSITIONS}" ]]; then
  echo "Missing transitions file: ${TRANSITIONS}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  echo "Run from the project root or set PYTHON=/path/to/.venv/bin/python" >&2
  exit 1
fi

echo "Launching latent predictor training on physical GPU ${GPU_ID}"
echo "Transitions: ${TRANSITIONS}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Log file:    ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${PYTHON}" -m lung_jepa.training.train_latent_predictor \
  --transitions "${TRANSITIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --val-fraction "${VAL_FRACTION}" \
  --test-fraction "${TEST_FRACTION}" \
  "${CLINICAL_FLAG}" \
  "${DELTA_FLAG}" \
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
echo "Started PID ${PID}"
echo "PID file: ${PID_FILE}"
echo "Follow logs with:"
echo "  tail -f ${LOG_FILE}"
