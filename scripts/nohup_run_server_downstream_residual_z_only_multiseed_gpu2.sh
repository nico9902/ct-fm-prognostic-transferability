#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
LAUNCH_LOG_FILE="${LAUNCH_LOG_FILE:-${LOG_DIR}/launch_downstream_${TAG}_residual_z_only_multiseed_gpu${GPU_ID}.log}"
LAUNCH_PID_FILE="${LAUNCH_PID_FILE:-${LOG_DIR}/launch_downstream_${TAG}_residual_z_only_multiseed_gpu${GPU_ID}.pid}"

mkdir -p "${LOG_DIR}"

nohup bash scripts/run_server_downstream_residual_z_only_multiseed_gpu2.sh > "${LAUNCH_LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${LAUNCH_PID_FILE}"
echo "Started residual multiseed downstream launcher PID ${PID}"
echo "Launcher PID file: ${LAUNCH_PID_FILE}"
echo "Launcher log: ${LAUNCH_LOG_FILE}"
