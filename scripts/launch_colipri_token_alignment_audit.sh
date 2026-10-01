#!/usr/bin/env bash
# Usage: bash scripts/launch_colipri_token_alignment_audit.sh GPU-uuid
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";cd "$ROOT"
GPU_UUID="${1:?Supply physical GPU UUID}";OUT=outputs/colipri_token_alignment_audit;mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock";flock -n 9 || { echo 'Another launcher is active';exit 1; }
if [[ -f "$OUT/pid" ]] && kill -0 "$(<"$OUT/pid")" 2>/dev/null;then echo 'Existing process alive';exit 1;fi
setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT/src:$ROOT/scripts" MPLCONFIGDIR=/tmp/colipri-mpl \
  "$ROOT/.venv/bin/python" -u scripts/run_colipri_token_alignment_audit.py \
  >> "$OUT/run.log" 2>&1 < /dev/null 9>&- &
PID_VALUE=$!;echo "$PID_VALUE" > "$OUT/pid";echo "Started PID $PID_VALUE on $GPU_UUID; log $OUT/run.log"
