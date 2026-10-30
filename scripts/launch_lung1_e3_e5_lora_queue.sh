#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_e3_e5_lora
exec 9>logs/lung1_e3_e5_lora/queue.lock
flock -n 9 || { echo 'Queue already running'; exit 1; }
export CUDA_VISIBLE_DEVICES=GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01
export TZ=Europe/Rome
export WANDB_MODE=offline
export MPLCONFIGDIR="$PWD/logs/lung1_e3_e5_lora/matplotlib"
echo "$$" > logs/lung1_e3_e5_lora/queue.pid
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > logs/lung1_e3_e5_lora/status; fi' EXIT
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "A100" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0), flush=True)'
for experiment in e3 e5; do
  echo "RUNNING $experiment $(date -Is)" | tee logs/lung1_e3_e5_lora/status
  bash "scripts/launch_lung1_${experiment}_lora.sh"
  echo "DONE $experiment $(date -Is)"
done
echo "COMPLETE $(date -Is)" | tee logs/lung1_e3_e5_lora/status
