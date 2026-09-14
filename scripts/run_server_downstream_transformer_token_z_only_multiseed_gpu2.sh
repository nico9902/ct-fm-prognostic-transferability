#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
TAG="${TAG:-pilot_500_balanced}"
GPU_ID="${GPU_ID:-2}"
TRANSITIONS="${TRANSITIONS:-${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_token_transitions.csv}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/latent_predictor_${TAG}_transformer_token_z_only_gpu${GPU_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/downstream_${TAG}_transformer_token_z_only_multiseed_gpu${GPU_ID}}"
OUTCOMES="${OUTCOMES:-cancer_label}"
SEEDS="${SEEDS:-0 1 2 3 4 42 101 202 303 404}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
CHECKPOINT="${CHECKPOINT:-}"

cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${ROOT_DIR}/src"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${TRANSITIONS}" ]]; then
  echo "Missing transitions file: ${TRANSITIONS}" >&2
  echo "Run scripts/run_server_extract_radjepa_tokens_pilot_500.sh and scripts/run_server_build_token_transitions_pilot_500.sh first." >&2
  exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Missing run directory: ${RUN_DIR}" >&2
    exit 1
  fi
  CHECKPOINT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'best-*.ckpt' | sort | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. CHECKPOINT=${CHECKPOINT} RUN_DIR=${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Transitions: ${TRANSITIONS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Device: ${DEVICE} (CUDA_VISIBLE_DEVICES=${GPU_ID})"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Seeds: ${SEEDS}"

for SEED in ${SEEDS}; do
  SEED_STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${SEED_STARTED}] Running transformer_token downstream seed ${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -u scripts/run_downstream_clinical_eval.py \
    --checkpoint "${CHECKPOINT}" \
    --transitions "${TRANSITIONS}" \
    --output-dir "${OUTPUT_ROOT}/seed_${SEED}" \
    --outcomes ${OUTCOMES} \
    --no-use-clinical \
    --no-use-delta \
    --prediction-mode direct \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --progress-every "${PROGRESS_EVERY}"
  SEED_FINISHED="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${SEED_FINISHED}] Finished transformer_token downstream seed ${SEED}"
done

"${PYTHON}" scripts/summarize_downstream_multiseed.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
