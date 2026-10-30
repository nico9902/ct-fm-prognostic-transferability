#!/usr/bin/env bash
# Phase 1 battery (CPU only), detached from the terminal: survives logout / closed laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/outputs/lung1_phase1_logs"
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore

if [[ "${1:-}" != "--run" ]]; then
  setsid nohup bash "$0" --run > "$LOGS/launcher.log" 2>&1 < /dev/null &
  echo "Phase 1 started (pid $!). Logs: $LOGS"
  exit 0
fi

cd "$ROOT/scripts"
PY="$ROOT/.venv/bin/python"
step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] start $name"
  if "$PY" "$@" > "$LOGS/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see $LOGS/$name.log)"
  fi
}
step ladder_primary        run_lung1_ceiling_ladder.py --workers 16
step ladder_radiomics_full run_lung1_ceiling_ladder.py --workers 16 --radiomics-full
step probing               run_lung1_embedding_probing.py --workers 24
step e4_no_refit           run_lung1_e4_no_refit_cox.py
echo "[$(date '+%F %T')] phase 1 finished"
touch "$LOGS/ALL_DONE"
