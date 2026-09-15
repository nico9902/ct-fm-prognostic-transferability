#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="${TAG:-pilot_1460_balanced}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
RUN_PREPARE="${RUN_PREPARE:-1}"

PROCESSED_DIR="${DATASET_ROOT}/processed/nlst/${TAG}"
SOURCE_MANIFEST="${PROCESSED_DIR}/nii_manifest.csv"
QC_DIR="${PROCESSED_DIR}/radjepa_nifti_qc_v2"
READY_MANIFEST="${QC_DIR}/nii_manifest_radjepa_ready.csv"
QC_REPORT="${QC_DIR}/nifti_qc_report.csv"
QC_SUMMARY="${QC_DIR}/nifti_qc_summary.json"
NPY_ROOT="${PROCESSED_DIR}/radjepa_npy_v2"
NPY_MANIFEST="${PROCESSED_DIR}/radjepa_npy_manifest_v2.csv"

cd "${ROOT_DIR}"

"${PYTHON}" scripts/validate_nlst_nifti_for_radjepa.py \
  --nifti-manifest "${SOURCE_MANIFEST}" \
  --output-ready-manifest "${READY_MANIFEST}" \
  --output-qc-report "${QC_REPORT}" \
  --output-summary "${QC_SUMMARY}"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  "${PYTHON}" scripts/prepare_radjepa_volumes.py \
    --nifti-manifest "${READY_MANIFEST}" \
    --output-root "${NPY_ROOT}" \
    --output-manifest "${NPY_MANIFEST}" \
    --require-qc-ready
fi

echo "Validated NIfTI manifest: ${READY_MANIFEST}"
echo "QC report: ${QC_REPORT}"
if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo "Canonical RadJEPA input manifest: ${NPY_MANIFEST}"
fi
