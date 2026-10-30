#!/usr/bin/env bash
# Phase 2.3-2.5 for the tumour-centred methods (Pai, Hosny), detached from the terminal.
# Requires the 2.2 extraction and its pre-specified check (Spearman >= 0.9 vs published Maastro).
# Hosny fine-tuning runs separately (run_lung1_hosny_finetune.py, V100 FP32).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/outputs/lung1_phase2_logs"
mkdir -p "$LOGS"

if [[ "${1:-}" != "--run" ]]; then
  setsid nohup bash "$0" --run > "$LOGS/launcher.log" 2>&1 < /dev/null &
  echo "Phase 2 started (pid $!). Logs: $LOGS"
  exit 0
fi

cd "$ROOT/scripts"
PY="$ROOT/.venv/bin/python"
"$PY" - <<'EOF'
import json, sys
from pathlib import Path
check = json.load(open("../outputs/lung1_e1_embeddings/hosny/verification.json"))
print(json.dumps(check, indent=2))
decision = Path("../outputs/lung1_e1_embeddings/hosny/verification_decision.json")
if check["spearman_p_dead_vs_published"] < 0.9 and not decision.exists():
    sys.exit("Pre-specified Hosny check failed (Spearman < 0.9): stop and investigate")
if decision.exists():
    print("Check failed; proceeding under the documented deviation:", decision.read_text())
EOF
export CUDA_DEVICE_ORDER=PCI_BUS_ID MPLCONFIGDIR="$LOGS/matplotlib" WANDB_CACHE_DIR="$LOGS/wandb_cache" WANDB_MODE=offline PYTHONWARNINGS=ignore
step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] start $name"
  if "$@" > "$LOGS/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see $LOGS/$name.log)"
  fi
}
heads() {  # BF16 heads on the A100, as for the whole-CT FMs
  for method in linear mlp; do
    for model in pai hosny; do
      step "${method}_${model}" env CUDA_VISIBLE_DEVICES="${HEADS_GPU:-3}" \
        "$PY" -u run_lung1_bf16_selected_heads.py --method "$method" --model "$model" --wandb-mode offline
    done
  done
}
heads &
step e3_pca16_pai   "$PY" -u run_lung1_e3_frozen_linear.py --components 16 --output-dir "$ROOT/outputs/lung1_e3_frozen_linear_pca16" --only-model pai --no-aggregate
step e3_pca16_hosny "$PY" -u run_lung1_e3_frozen_linear.py --components 16 --output-dir "$ROOT/outputs/lung1_e3_frozen_linear_pca16" --only-model hosny
step ladder_tumour_centred "$PY" -u run_lung1_ceiling_ladder.py --workers 16 --tumor-centred
step e3_pca8_pai    "$PY" -u run_lung1_e3_frozen_linear.py --components 8 --only-model pai --no-aggregate
step e3_pca8_hosny  "$PY" -u run_lung1_e3_frozen_linear.py --components 8 --only-model hosny
wait
echo "[$(date '+%F %T')] phase 2 finished"
touch "$LOGS/ALL_DONE"
