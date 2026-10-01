#!/usr/bin/env bash
# Usage: bash scripts/launch_colipri_prognostic_adapter.sh GPU-uuid [--smoke]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU_UUID="${1:?Supply physical GPU UUID}"
EXTRA=()
OUT=outputs/colipri_prognostic_adapter
if [[ "${2:-}" == '--smoke' ]]; then
  EXTRA=(--smoke)
  OUT=outputs/colipri_prognostic_adapter_smoke
elif [[ -n "${2:-}" ]]; then
  echo 'Only --smoke is supported as second argument'; exit 1
fi
mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo 'Another launcher is active'; exit 1; }
if [[ -f "$OUT/pid" ]] && kill -0 "$(<"$OUT/pid")" 2>/dev/null; then
  echo 'Existing battery process still alive'; exit 1
fi
setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$ROOT/src:$ROOT/scripts" MPLCONFIGDIR=/tmp/colipri-mpl \
  "$ROOT/.venv/bin/python" -u scripts/run_colipri_prognostic_adapter.py \
  "${EXTRA[@]}" >> "$OUT/run.log" 2>&1 < /dev/null 9>&- &
BATTERY_PID=$!
echo "$BATTERY_PID" > "$OUT/pid"
echo "Started PID $BATTERY_PID on $GPU_UUID; log $OUT/run.log"
