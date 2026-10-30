#!/usr/bin/env bash
# Nested-CV final refits for the MLP and SQA readouts on the whole development set of each
# outer fold, for the epochs chosen by scripts/select_lung1_nested_heads.py
# (docs/lung1_sweep_validation_protocol.md).
set -euo pipefail
cd "$(dirname "$0")/.."
out="outputs/lung1_nested/heads"; logs="logs/lung1_nested/heads"
mkdir -p "$logs" "$out/final"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01}"
export MPLCONFIGDIR="$PWD/$logs/matplotlib"
trap 'code=$?; if ((code != 0)); then echo "FAILED exit=$code $(date -Is)" > "$out/FAILED_final"; fi' EXIT
rm -f "$out/FAILED_final" "$out/COMPLETE_final"
.venv/bin/python scripts/select_lung1_nested_heads.py
for method in mlp sqa; do
  if [[ "$method" == mlp ]]; then models=(colipri tangerine radjepa ctclip dinov3 pai hosny)
  else models=(colipri tangerine radjepa ctclip dinov3); fi
  for model in "${models[@]}"; do
    tag="final_${method}_${model}"
    epochs=$(.venv/bin/python -c "import json,sys; s=json.load(open('$out/final/selected_epochs.json'))['$method']['$model']; print(json.dumps({k: v['epochs'] for k, v in s.items()}))")
    echo "START $tag $epochs $(date -Is)"
    .venv/bin/python -u scripts/run_lung1_bf16_selected_heads.py \
      --method "$method" --model "$model" --fixed-trainval --fold-epochs "$epochs" \
      --wandb-mode offline --output-root "$out/final" > "$logs/${tag}.log" 2>&1
    echo "DONE $tag $(date -Is)"
  done
done
echo "COMPLETE $(date -Is)" > "$out/COMPLETE_final"
