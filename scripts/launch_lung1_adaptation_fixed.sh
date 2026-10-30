#!/usr/bin/env bash
# Fixed adaptation configuration for the encoders not in the preliminary fixed-epoch sweep
# (docs/lung1_sweep_validation_protocol.md, amendment 2): LoRA lr 1e-3 and partial lr 1e-4,
# 20 epochs, whole development set, final epoch. COLIPRI/TANGERINE reuse outputs/lung1_phase3/.
set -euo pipefail
cd "$(dirname "$0")/.."
out=outputs/lung1_adaptation_fixed; logs=logs/lung1_adaptation_fixed
mkdir -p "$out" "$logs"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
export MPLCONFIGDIR="$PWD/$logs/matplotlib"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$out/FAILED"; fi' EXIT
rm -f "$out/FAILED" "$out/COMPLETE"
for model in radjepa dinov3 ctclip; do
  py=.venv/bin/python; [[ "$model" == dinov3 ]] && py=.venv-dinov3/bin/python
  for spec in lora:1e-3 partial:1e-4; do
    method="${spec%%:*}"; rate="${spec##*:}"; tag="${method}_${model}_lr${rate}_ep20"
    echo "START $tag $(date -Is)"
    "$py" -u scripts/run_lung1_e2_encoder_bf16_lightning.py \
      --method "$method" --model "$model" --learning-rate "$rate" --epochs 20 \
      --fixed-trainval --wandb-mode offline --output-root "$out/lr${rate}_ep20" \
      > "$logs/${tag}.log" 2>&1
    echo "DONE $tag $(date -Is)"
  done
done
echo "COMPLETE $(date -Is)" > "$out/COMPLETE"
