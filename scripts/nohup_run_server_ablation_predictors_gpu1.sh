#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
LAUNCH_LOG_FILE="${LAUNCH_LOG_FILE:-${LOG_DIR}/launch_ablation_predictors_${TAG}_gpu${GPU_ID}.log}"
LAUNCH_PID_FILE="${LAUNCH_PID_FILE:-${LOG_DIR}/launch_ablation_predictors_${TAG}_gpu${GPU_ID}.pid}"

mkdir -p "${LOG_DIR}"

nohup bash scripts/run_server_ablation_predictors_gpu1.sh > "${LAUNCH_LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${LAUNCH_PID_FILE}"
echo "Started ablation launcher PID ${PID}"
echo "Launcher PID file: ${LAUNCH_PID_FILE}"
echo "Launcher log: ${LAUNCH_LOG_FILE}"
echo "Summary will be written to:"
echo "  ${OUTPUT_ROOT}/ablation_${TAG}_gpu${GPU_ID}/ablation_summary.xlsx"
