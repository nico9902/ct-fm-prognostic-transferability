#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="${TAG:-pilot_2000_balanced}"
N_PATIENTS="${N_PATIENTS:-2000}"
SEED="${SEED:-42}"
MIN_TIMEPOINTS="${MIN_TIMEPOINTS:-2}"
MIN_IMAGES="${MIN_IMAGES:-50}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATASET_ROOT}/manifests}"
DICOM_DIR="${DICOM_DIR:-${DATASET_ROOT}/raw/nlst/dicom}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATASET_ROOT}/processed/nlst/${TAG}}"
CLINICAL_DIR="${CLINICAL_DIR:-${DATASET_ROOT}/metadata/nlst/clinical}"
RUN_BUILD_DOWNLOAD="${RUN_BUILD_DOWNLOAD:-1}"

cd "${ROOT_DIR}"
mkdir -p "${MANIFEST_DIR}" "${DICOM_DIR}" "${PROCESSED_DIR}" "${CLINICAL_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  echo "Set PYTHON=/path/to/python or activate/create the project venv." >&2
  exit 1
fi

echo "Running NLST pilot pipeline"
echo "TAG=${TAG}"
echo "N_PATIENTS=${N_PATIENTS}"
echo "SEED=${SEED}"
echo "MIN_TIMEPOINTS=${MIN_TIMEPOINTS}"
echo "MIN_IMAGES=${MIN_IMAGES}"
echo "PYTHON=${PYTHON}"

if [[ "${RUN_BUILD_DOWNLOAD}" == "1" ]]; then
  "${PYTHON}" scripts/build_nlst_manifest.py \
    --max-patients "${N_PATIENTS}" \
    --seed "${SEED}" \
    --min-timepoints "${MIN_TIMEPOINTS}" \
    --min-images "${MIN_IMAGES}" \
    --cohort-mode cancer-balanced \
    --tag "${TAG}" \
    --output-dir "${MANIFEST_DIR}" \
    --download-dir "${DICOM_DIR}" \
    --clinical-output-dir "${CLINICAL_DIR}" \
    --download \
    --download-clinical
fi

SERIES_MANIFEST="${MANIFEST_DIR}/nlst_${TAG}_series.csv"
if [[ ! -f "${SERIES_MANIFEST}" ]]; then
  echo "Missing series manifest: ${SERIES_MANIFEST}" >&2
  echo "Set RUN_BUILD_DOWNLOAD=1 or MANIFEST_DIR to the directory containing nlst_${TAG}_series.csv." >&2
  exit 1
fi

"${PYTHON}" scripts/convert_nlst_dicom_to_nifti.py \
  --manifest "${SERIES_MANIFEST}" \
  --dicom-root "${DICOM_DIR}" \
  --output-root "${PROCESSED_DIR}/nii" \
  --output-manifest "${PROCESSED_DIR}/nii_manifest.csv" \
  --on-error skip

"${PYTHON}" scripts/prepare_radjepa_volumes.py \
  --nifti-manifest "${PROCESSED_DIR}/nii_manifest.csv" \
  --output-root "${PROCESSED_DIR}/radjepa_npy" \
  --output-manifest "${PROCESSED_DIR}/radjepa_npy_manifest.csv"
