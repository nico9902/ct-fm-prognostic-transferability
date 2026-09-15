#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_2000_balanced}"
EXPECTED_PATIENTS="${EXPECTED_PATIENTS:-2000}"
EXPECTED_PER_CLASS="${EXPECTED_PER_CLASS:-}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
SEED="${SEED:-42}"

PAIRS="${PAIRS:-${DATASET_ROOT}/manifests/nlst_${TAG}_pairs.csv}"
SUBJECTS="${SUBJECTS:-${DATASET_ROOT}/manifests/nlst_${TAG}_subjects.csv}"
EMBEDDINGS="${EMBEDDINGS:-${DATASET_ROOT}/embeddings/radjepa3d_${TAG}_tokens_manifest.csv}"
CLINICAL_PRSN="${CLINICAL_PRSN:-${DATASET_ROOT}/metadata/nlst/clinical/cohort_subsets/${TAG}/nlst_prsn.csv}"
OUTPUT="${OUTPUT:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_token_transitions.csv}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

"${PYTHON}" scripts/build_latent_transition_dataset.py \
  --pairs "${PAIRS}" \
  --embeddings "${EMBEDDINGS}" \
  --clinical-prsn "${CLINICAL_PRSN}" \
  --output "${OUTPUT}"

VALIDATE_ARGS=(
  --subjects "${SUBJECTS}" \
  --transitions "${OUTPUT}" \
  --expected-patients "${EXPECTED_PATIENTS}" \
  --val-fraction "${VAL_FRACTION}" \
  --test-fraction "${TEST_FRACTION}" \
  --seed "${SEED}"
)
if [[ -n "${EXPECTED_PER_CLASS}" ]]; then
  VALIDATE_ARGS+=(--expected-per-class "${EXPECTED_PER_CLASS}")
fi

"${PYTHON}" scripts/validate_nlst_cohort.py "${VALIDATE_ARGS[@]}"
