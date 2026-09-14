#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"

INPUT_MANIFEST="${INPUT_MANIFEST:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_npy_manifest.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/embeddings/radjepa3d_${TAG}_tokens}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${DATASET_ROOT}/embeddings/radjepa3d_${TAG}_tokens_manifest.csv}"
BATCH_SIZE="${BATCH_SIZE:-1}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/extract_radjepa_embeddings.py \
  --input-manifest "${INPUT_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --output-mode tokens
