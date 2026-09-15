#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TAG="${TAG:-pilot_1460_balanced}"
export N_PATIENTS="${N_PATIENTS:-1460}"
export EXPECTED_PATIENTS="${EXPECTED_PATIENTS:-1460}"
export EXPECTED_PER_CLASS="${EXPECTED_PER_CLASS:-730}"
export GPU_ID="${GPU_ID:-1}"
export PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
export DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"

RUN_CONVERT_PREPARE="${RUN_CONVERT_PREPARE:-1}"
RUN_EXTRACT_TOKENS="${RUN_EXTRACT_TOKENS:-1}"
RUN_BUILD_TRANSITIONS="${RUN_BUILD_TRANSITIONS:-1}"
ALLOW_CONCURRENT="${ALLOW_CONCURRENT:-0}"

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi

echo "Resume preprocessing"
echo "TAG=${TAG}"
echo "N_PATIENTS=${N_PATIENTS}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "GPU_ID=${GPU_ID}"
echo "PYTHON=${PYTHON}"
echo "RUN_CONVERT_PREPARE=${RUN_CONVERT_PREPARE}"
echo "RUN_EXTRACT_TOKENS=${RUN_EXTRACT_TOKENS}"
echo "RUN_BUILD_TRANSITIONS=${RUN_BUILD_TRANSITIONS}"

if [[ "${ALLOW_CONCURRENT}" != "1" ]]; then
  if pgrep -af "convert_nlst_dicom_to_nifti.py.*${TAG}" >/dev/null; then
    echo "A conversion process for ${TAG} is already running:" >&2
    pgrep -af "convert_nlst_dicom_to_nifti.py.*${TAG}" >&2
    echo "Wait for it to finish, or rerun with ALLOW_CONCURRENT=1 if you are sure." >&2
    exit 1
  fi
fi

if [[ "${RUN_CONVERT_PREPARE}" == "1" ]]; then
  echo
  echo "Step 1/3: resume DICOM -> NIfTI and RadJEPA volume preparation"
  export RUN_BUILD_DOWNLOAD=0
  bash "${ROOT_DIR}/scripts/run_server_pilot_max_balanced.sh"
fi

if [[ "${RUN_EXTRACT_TOKENS}" == "1" ]]; then
  echo
  echo "Step 2/3: extract RadJEPA token embeddings"
  bash "${ROOT_DIR}/scripts/run_server_extract_radjepa_tokens_pilot_max_balanced.sh"
fi

if [[ "${RUN_BUILD_TRANSITIONS}" == "1" ]]; then
  echo
  echo "Step 3/3: build token transitions and validate cohort"
  bash "${ROOT_DIR}/scripts/run_server_build_token_transitions_pilot_max_balanced.sh"
fi

echo
echo "Resume preprocessing completed."
echo "Next training command:"
echo "  bash scripts/run_server_train_transformer_token_cls_z_only_pilot_max_balanced_gpu1.sh"
