#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"

PAIRS="${PAIRS:-${DATASET_ROOT}/manifests/nlst_${TAG}_pairs.csv}"
EMBEDDINGS="${EMBEDDINGS:-${DATASET_ROOT}/embeddings/radjepa3d_${TAG}_tokens_manifest.csv}"
CLINICAL_PRSN="${CLINICAL_PRSN:-${DATASET_ROOT}/metadata/nlst/clinical/cohort_subsets/${TAG}/nlst_prsn.csv}"
OUTPUT="${OUTPUT:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_token_transitions.csv}"

"${PYTHON}" scripts/build_latent_transition_dataset.py \
  --pairs "${PAIRS}" \
  --embeddings "${EMBEDDINGS}" \
  --clinical-prsn "${CLINICAL_PRSN}" \
  --output "${OUTPUT}"
