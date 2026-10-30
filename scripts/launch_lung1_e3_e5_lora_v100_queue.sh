#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_e3_e5_lora_v100
exec 9>logs/lung1_e3_e5_lora_v100/queue.lock
flock -n 9 || { echo 'V100 queue already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=GPU-70cb4254-c439-93d4-1f5b-fbfe9d8e3d0c
export TZ=Europe/Rome WANDB_MODE=offline
export RUN_PRECISION=16-mixed
export MPLCONFIGDIR="$PWD/logs/lung1_e3_e5_lora_v100/matplotlib"
echo "$$" > logs/lung1_e3_e5_lora_v100/queue.pid
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > logs/lung1_e3_e5_lora_v100/status; fi' EXIT
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and "V100" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0), flush=True)'
for experiment in e3 e5; do
  export RUN_OUTPUT="outputs/lung1_${experiment}_lora_v100_fp16"
  export RUN_LOGS="logs/lung1_${experiment}_lora_v100_fp16"
  echo "RUNNING $experiment $(date -Is)" | tee logs/lung1_e3_e5_lora_v100/status
  bash "scripts/launch_lung1_${experiment}_lora.sh"
  echo "DONE $experiment $(date -Is)"
done
echo "COMPLETE $(date -Is)" | tee logs/lung1_e3_e5_lora_v100/status
