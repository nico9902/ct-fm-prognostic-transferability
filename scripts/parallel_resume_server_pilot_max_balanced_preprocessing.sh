#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TAG="${TAG:-pilot_1460_balanced}"
export N_PATIENTS="${N_PATIENTS:-1460}"
export EXPECTED_PATIENTS="${EXPECTED_PATIENTS:-1460}"
export EXPECTED_PER_CLASS="${EXPECTED_PER_CLASS:-730}"
export GPU_ID="${GPU_ID:-1}"
export JOBS="${JOBS:-4}"
export PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
export DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"

RUN_NIFTI="${RUN_NIFTI:-1}"
RUN_PREPARE_NPY="${RUN_PREPARE_NPY:-1}"
RUN_EXTRACT_TOKENS="${RUN_EXTRACT_TOKENS:-1}"
RUN_BUILD_TRANSITIONS="${RUN_BUILD_TRANSITIONS:-1}"
KILL_EXISTING="${KILL_EXISTING:-1}"

SERIES_MANIFEST="${DATASET_ROOT}/manifests/nlst_${TAG}_series.csv"
PROCESSED_DIR="${DATASET_ROOT}/processed/nlst/${TAG}"
NII_MANIFEST="${PROCESSED_DIR}/nii_manifest.csv"
NPY_ROOT="${PROCESSED_DIR}/radjepa_npy"
NPY_MANIFEST="${PROCESSED_DIR}/radjepa_npy_manifest.csv"

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi

echo "Parallel resume preprocessing"
echo "TAG=${TAG}"
echo "N_PATIENTS=${N_PATIENTS}"
echo "JOBS=${JOBS}"
echo "GPU_ID=${GPU_ID}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "PYTHON=${PYTHON}"

if [[ "${KILL_EXISTING}" == "1" ]]; then
  PATTERNS=(
    "scripts/resume_server_pilot_max_balanced_preprocessing.sh"
    "scripts/run_server_pilot_max_balanced.sh"
    "scripts/run_server_pilot_2000_balanced.sh"
    "convert_nlst_dicom_to_nifti.py.*${TAG}"
    "extract_radjepa_embeddings.py.*${TAG}"
    "build_latent_transition_dataset.py.*${TAG}"
  )
  PIDS=""
  for PATTERN in "${PATTERNS[@]}"; do
    MATCHES="$(pgrep -f "${PATTERN}" || true)"
    if [[ -n "${MATCHES}" ]]; then
      PIDS="${PIDS}"$'\n'"${MATCHES}"
    fi
  done
  PIDS="$(printf '%s\n' "${PIDS}" | awk -v self="$$" -v bashpid="${BASHPID}" 'NF && $1 != self && $1 != bashpid {print $1}' | sort -u)"
  if [[ -n "${PIDS}" ]]; then
    echo "Stopping existing ${TAG} preprocessing processes: ${PIDS}"
    kill ${PIDS} || true
    sleep 5
  fi
  REMAINING=""
  for PATTERN in "${PATTERNS[@]}"; do
    MATCHES="$(pgrep -f "${PATTERN}" || true)"
    if [[ -n "${MATCHES}" ]]; then
      REMAINING="${REMAINING}"$'\n'"${MATCHES}"
    fi
  done
  REMAINING="$(printf '%s\n' "${REMAINING}" | awk -v self="$$" -v bashpid="${BASHPID}" 'NF && $1 != self && $1 != bashpid {print $1}' | sort -u)"
  if [[ -n "${REMAINING}" ]]; then
    echo "Force stopping remaining ${TAG} preprocessing processes: ${REMAINING}"
    kill -9 ${REMAINING} || true
    sleep 2
  fi
fi

if [[ "${RUN_NIFTI}" == "1" ]]; then
  echo
  echo "Step 1/4: parallel DICOM -> NIfTI resume"
  KILL_EXISTING=0 bash "${ROOT_DIR}/scripts/parallel_resume_nifti_pilot_max_balanced.sh"
fi

if [[ "${RUN_PREPARE_NPY}" == "1" ]]; then
  echo
  echo "Step 2/4: prepare RadJEPA .npy volumes"
  "${PYTHON}" scripts/prepare_radjepa_volumes.py \
    --nifti-manifest "${NII_MANIFEST}" \
    --output-root "${NPY_ROOT}" \
    --output-manifest "${NPY_MANIFEST}"
fi

if [[ "${RUN_EXTRACT_TOKENS}" == "1" ]]; then
  echo
  echo "Step 3/4: extract RadJEPA token embeddings"
  bash "${ROOT_DIR}/scripts/run_server_extract_radjepa_tokens_pilot_max_balanced.sh"
fi

if [[ "${RUN_BUILD_TRANSITIONS}" == "1" ]]; then
  echo
  echo "Step 4/4: build token transitions and validate cohort"
  bash "${ROOT_DIR}/scripts/run_server_build_token_transitions_pilot_max_balanced.sh"
fi

echo
echo "Parallel resume preprocessing completed."
echo "Next training command:"
echo "  bash scripts/run_server_train_transformer_token_cls_z_only_pilot_max_balanced_gpu1.sh"
