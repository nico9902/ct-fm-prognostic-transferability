#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LAUNCH_LOG_FILE="${LAUNCH_LOG_FILE:-${LOG_DIR}/launch_train_latent_predictor_${TAG}_gpu${GPU_ID}.log}"
LAUNCH_PID_FILE="${LAUNCH_PID_FILE:-${LOG_DIR}/launch_train_latent_predictor_${TAG}_gpu${GPU_ID}.pid}"

mkdir -p "${LOG_DIR}"

nohup bash scripts/run_server_train_predictor_gpu1.sh > "${LAUNCH_LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${LAUNCH_PID_FILE}"
echo "Started nohup launcher PID ${PID}"
echo "Launcher PID file: ${LAUNCH_PID_FILE}"
echo "Launcher log: ${LAUNCH_LOG_FILE}"
echo "Training log:"
echo "  ${LOG_DIR}/train_latent_predictor_${TAG}_gpu${GPU_ID}.log"
