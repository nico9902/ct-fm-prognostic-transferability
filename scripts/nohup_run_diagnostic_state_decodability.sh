#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"

LABELS="/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_labels/nlst_scan_diagnostic_labels.parquet"
NPY_MANIFEST="/ssd2/domenico/datasets/nlst/processed/nlst/pilot_1000_balanced/sliding_npy_manifest.csv"
OUTPUT_DIR="/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/diagnostic_decodability_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/diagnostic_decodability.pid"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${LABELS}" ]]; then
  echo "Missing labels: ${LABELS}" >&2
  exit 1
fi
if [[ ! -f "${NPY_MANIFEST}" ]]; then
  echo "Missing NPY manifest: ${NPY_MANIFEST}" >&2
  exit 1
fi

WANDB_ARGS=()
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  WANDB_ARGS+=(--wandb)
fi

cd "${REPO_DIR}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nohup env CUDA_VISIBLE_DEVICES=0 GPU_DEVICE=0 CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG}" "${PYTHON_BIN}" \
  scripts/run_diagnostic_state_decodability.py \
  --labels "${LABELS}" \
  --npy-manifest "${NPY_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-8}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --patience "${PATIENCE:-15}" \
  "${WANDB_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Started diagnostic decodability experiments"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
