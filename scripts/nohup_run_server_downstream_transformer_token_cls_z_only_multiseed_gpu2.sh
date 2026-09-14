#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/logs}"
LAUNCH_LOG_FILE="${LAUNCH_LOG_FILE:-${LOG_DIR}/launch_downstream_${TAG}_transformer_token_cls_z_only_multiseed_gpu${GPU_ID}.log}"
LAUNCH_PID_FILE="${LAUNCH_PID_FILE:-${LOG_DIR}/launch_downstream_${TAG}_transformer_token_cls_z_only_multiseed_gpu${GPU_ID}.pid}"

mkdir -p "${LOG_DIR}"

export PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
export DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
export TAG
export GPU_ID
export TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_token_transitions.csv}"
export RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/latent_predictor_${TAG}_transformer_token_cls_z_only_gpu${GPU_ID}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/downstream_${TAG}_transformer_token_cls_z_only_multiseed_gpu${GPU_ID}}"
export OUTCOMES="${OUTCOMES:-cancer_label}"
export SEEDS="${SEEDS:-0 1 2 3 4 42 101 202 303 404}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export DEVICE="${DEVICE:-cuda}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
export CHECKPOINT="${CHECKPOINT:-}"

nohup bash "${SCRIPT_DIR}/run_server_downstream_transformer_token_cls_z_only_multiseed_gpu2.sh" > "${LAUNCH_LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${LAUNCH_PID_FILE}"
echo "Started transformer_token_cls multiseed downstream launcher PID ${PID}"
echo "Launcher PID file: ${LAUNCH_PID_FILE}"
echo "Launcher log: ${LAUNCH_LOG_FILE}"
