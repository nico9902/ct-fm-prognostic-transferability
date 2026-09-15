#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"

TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_transitions.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-32-true}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
WANDB_PROJECT="${WANDB_PROJECT:-lung-jepa-world-model}"
WANDB_MODE_FLAG="${WANDB_MODE_FLAG:---wandb}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

if [[ ! -f "${TRANSITIONS}" ]]; then
  echo "Missing transitions file: ${TRANSITIONS}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  echo "Run from the project root or set PYTHON=/path/to/.venv/bin/python" >&2
  exit 1
fi

run_experiment() {
  local name="$1"
  local clinical_flag="$2"
  local delta_flag="$3"
  local output_dir="${OUTPUT_ROOT}/latent_predictor_${TAG}_${name}_gpu${GPU_ID}"
  local log_file="${LOG_DIR}/train_latent_predictor_${TAG}_${name}_gpu${GPU_ID}.log"
  local wandb_name="radjepa3d_${TAG}_${name}_gpu${GPU_ID}"

  echo "============================================================"
  echo "Experiment: ${name}"
  echo "Output:     ${output_dir}"
  echo "Log:        ${log_file}"
  echo "GPU:        ${GPU_ID}"
  echo "Clinical:   ${clinical_flag}"
  echo "Delta:      ${delta_flag}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -m lung_jepa.training.train_latent_predictor \
    --transitions "${TRANSITIONS}" \
    --output-dir "${output_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --val-fraction "${VAL_FRACTION}" \
    --test-fraction "${TEST_FRACTION}" \
    "${clinical_flag}" \
    "${delta_flag}" \
    --test \
    --accelerator gpu \
    --devices 1 \
    --precision "${PRECISION}" \
    "${WANDB_MODE_FLAG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${wandb_name}" \
    2>&1 | tee "${log_file}"
}

run_experiment "full" "--use-clinical" "--use-delta"
run_experiment "no_clinical" "--no-use-clinical" "--use-delta"
run_experiment "no_delta" "--use-clinical" "--no-use-delta"
run_experiment "z_only" "--no-use-clinical" "--no-use-delta"

"${PYTHON}" scripts/summarize_ablation_reports.py \
  --run-root "${OUTPUT_ROOT}" \
  --pattern "latent_predictor_${TAG}_*_gpu${GPU_ID}" \
  --output-dir "${OUTPUT_ROOT}/ablation_${TAG}_gpu${GPU_ID}"
