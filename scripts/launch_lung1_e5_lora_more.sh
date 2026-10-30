#!/usr/bin/env bash
# E5 exploratory LoRA for DINOv3, Rad-JEPA-3D and CT-CLIP on the A100 in BF16 (as the internal
# adaptation runs); fixed configuration lr 1e-3, 20 epochs, trained on all LUNG1 patients.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline TZ=Europe/Rome
export CUDA_VISIBLE_DEVICES=GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01
out=outputs/lung1_e5_lora_a100_bf16; logs=logs/lung1_e5_lora_a100_bf16
export MPLCONFIGDIR="$PWD/$logs/matplotlib"
mkdir -p "$out" "$logs"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$out/FAILED"; fi' EXIT
rm -f "$out/FAILED" "$out/COMPLETE"
for model in radjepa dinov3 ctclip; do
  py=.venv/bin/python; [[ "$model" == dinov3 ]] && py=.venv-dinov3/bin/python
  for stage in train cache predict evaluate; do
    echo "START $stage $model $(date -Is)" | tee -a "$logs/status"
    if [[ "$stage" == cache ]]; then
      "$py" -u scripts/cache_lung1_e2_inputs.py --model "$model" \
        --cohort data/derived/radiogenomics/cohort_with_gtv_volume.csv \
        --status-column global_status --clamp-min-hu -1024 \
        --output-root outputs/radiogenomics_lora_inputs > "$logs/cache_$model.log" 2>&1
    else
      "$py" -u scripts/run_lung1_e5_lora_more.py --precision bf16-mixed --output-root "$out" \
        --model "$model" --stage "$stage" > "$logs/${stage}_$model.log" 2>&1
    fi
    echo "DONE $stage $model $(date -Is)" | tee -a "$logs/status"
  done
done
date -Is > "$out/COMPLETE"
