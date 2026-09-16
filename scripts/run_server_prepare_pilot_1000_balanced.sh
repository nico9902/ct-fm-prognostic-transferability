#!/usr/bin/env bash
# Rebuild only the fixed local cohort; never resample or download patients.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data/nlst}"
TAG=pilot_1000_balanced
PROCESSED_DIR="${DATASET_ROOT}/processed/nlst/${TAG}"
mkdir -p "${PROCESSED_DIR}"
"${PYTHON}" scripts/convert_nlst_dicom_to_nifti.py \
  --manifest "${DATASET_ROOT}/manifests/nlst_${TAG}_series.csv" \
  --dicom-root "${DATASET_ROOT}/raw/nlst/dicom" \
  --output-root "${PROCESSED_DIR}/nii" \
  --output-manifest "${PROCESSED_DIR}/nii_manifest.csv" --on-error skip
"${PYTHON}" scripts/validate_nlst_nifti_for_radjepa.py \
  --nifti-manifest "${PROCESSED_DIR}/nii_manifest.csv" \
  --output-ready-manifest "${PROCESSED_DIR}/nii_manifest_ready.csv" \
  --output-qc-report "${PROCESSED_DIR}/nifti_qc_report.csv" \
  --output-summary "${PROCESSED_DIR}/nifti_qc_summary.json"
"${PYTHON}" scripts/prepare_radjepa_volumes.py \
  --nifti-manifest "${PROCESSED_DIR}/nii_manifest_ready.csv" \
  --output-root "${PROCESSED_DIR}/radjepa_npy" \
  --output-manifest "${PROCESSED_DIR}/radjepa_npy_manifest.csv" --require-qc-ready
