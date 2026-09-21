#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OUT="${OUT_DIR:-outputs/colipri_temporal_adapter_ablation}"; mkdir -p "$OUT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="python3"; fi
nohup env PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PYTHON_BIN" scripts/run_colipri_temporal_adapter_ablation.py --output-dir "$OUT" --device "${DEVICE:-auto}" --wandb-project "${WANDB_PROJECT:-nlst-colipri-temporal-adapter}" > "$OUT/run.log" 2>&1 < /dev/null &
echo $! > "$OUT/pid"; echo "Started PID $(cat "$OUT/pid"). Log: $OUT/run.log"
