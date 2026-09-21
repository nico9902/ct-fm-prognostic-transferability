#!/usr/bin/env bash
# Usage: bash scripts/launch_colipri_preserving_adapter.sh GPU-uuid [config] [output]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU_UUID="${1:?Supply a physical GPU UUID from nvidia-smi}"
CONFIG="${2:-configs/model/colipri_preserving_adapter.yaml}"
OUT="${3:-outputs/colipri_preserving_adapter}"
PYTHON_BIN="$ROOT/.venv/bin/python"
mkdir -p "$OUT"
# Serialize duplicate launch checks. Keep prior logs; Python resumes completed units.
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo 'Another launcher is active'; exit 1; }
if [ -f "$OUT/pid" ] && kill -0 "$(<"$OUT/pid")" 2>/dev/null; then
  echo "Process $(<"$OUT/pid") is already running"
  exit 1
fi
setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$ROOT/src:$ROOT/scripts" MPLCONFIGDIR=/tmp/colipri-mpl \
  "$PYTHON_BIN" -u scripts/run_colipri_preserving_adapter.py \
  --config "$CONFIG" --output-dir "$OUT" >> "$OUT/run.log" 2>&1 < /dev/null 9>&- &
BATTERY_PID=$!
echo "$BATTERY_PID" > "$OUT/pid"
echo "Started PID $BATTERY_PID on $GPU_UUID; log $OUT/run.log"
