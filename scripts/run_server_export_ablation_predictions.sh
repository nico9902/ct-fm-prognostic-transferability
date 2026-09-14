#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"
TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_transitions.csv}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

export_one() {
  local name="$1"
  local clinical_flag="$2"
  local delta_flag="$3"
  local output_dir="outputs/latent_predictor_${TAG}_${name}_gpu${GPU_ID}"
  local checkpoints=("${output_dir}"/best-*.ckpt)
  if [[ ! -f "${checkpoints[0]}" ]]; then
    echo "No checkpoint found for ${name}: ${output_dir}" >&2
    return 1
  fi
  local checkpoint="${checkpoints[0]}"

  "${PYTHON}" scripts/export_test_predictions_from_checkpoint.py \
    --checkpoint "${checkpoint}" \
    --transitions "${TRANSITIONS}" \
    --output-dir "${output_dir}" \
    "${clinical_flag}" \
    "${delta_flag}" \
    --batch-size 32 \
    --num-workers 4
}

export_one "full" "--use-clinical" "--use-delta"
export_one "no_clinical" "--no-use-clinical" "--use-delta"
export_one "no_delta" "--use-clinical" "--no-use-delta"
export_one "z_only" "--no-use-clinical" "--no-use-delta"
