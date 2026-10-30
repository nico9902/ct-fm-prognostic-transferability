#!/usr/bin/env bash
# E5 only (external LoRA) on V100 FP16, after the E3 LoRA queue was cancelled on 26 October 2026.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_e3_e5_lora_v100
export CUDA_VISIBLE_DEVICES=GPU-70cb4254-c439-93d4-1f5b-fbfe9d8e3d0c
export TZ=Europe/Rome WANDB_MODE=offline
export RUN_PRECISION=16-mixed
export RUN_OUTPUT=outputs/lung1_e5_lora_v100_fp16
export RUN_LOGS=logs/lung1_e5_lora_v100_fp16
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and "V100" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0), flush=True)'
echo "RUNNING e5 $(date -Is)" | tee -a logs/lung1_e3_e5_lora_v100/status
bash scripts/launch_lung1_e5_lora.sh
echo "DONE e5 $(date -Is)" | tee -a logs/lung1_e3_e5_lora_v100/status
