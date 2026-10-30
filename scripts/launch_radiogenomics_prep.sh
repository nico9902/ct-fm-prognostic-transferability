#!/usr/bin/env bash
# Phase 4.1-4.2, outcome-blind: Radiogenomics thorax cache + GTV checks, radiomics, Pai/Hosny
# (GPU 0) and the five FM embeddings on all eligible CTs (A100), detached from the terminal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/outputs/radiogenomics_logs"
mkdir -p "$LOGS"

if [[ "${1:-}" != "--run" ]]; then
  setsid nohup bash "$0" --run > "$LOGS/launcher.log" 2>&1 < /dev/null &
  echo "Radiogenomics preparation started (pid $!). Logs: $LOGS"
  exit 0
fi

cd "$ROOT/scripts"
PY="$ROOT/.venv/bin/python"
LUNGMASK_PY=/home/domenico/lung_ct_3d2d_synthesis/myenv/bin/python  # same environment as the LUNG1 cache
COHORT="$ROOT/data/derived/radiogenomics/cohort.csv"
CHECKED="$ROOT/data/derived/radiogenomics/cohort_with_gtv_volume.csv"
CACHE="$ROOT/data/processed/radiogenomics/thorax_hu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID MPLCONFIGDIR="$LOGS/matplotlib" PYTHONWARNINGS=ignore
step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] start $name"
  if "$@" > "$LOGS/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see $LOGS/$name.log)"
    return 1
  fi
}
tumour_track() {
  step cache env CUDA_VISIBLE_DEVICES=0 "$LUNGMASK_PY" -u preprocess_radiogenomics.py --workers 4
  step radiomics "$ROOT/outputs/lung1_e3_baselines/venv_radiomics/bin/python" -u extract_lung1_e3_radiomics.py --cohort "$CHECKED" --input "$CACHE" --workers 16 \
    --output "$ROOT/outputs/radiogenomics_embeddings/radiomics_features.csv"
  step pai_hosny env CUDA_VISIBLE_DEVICES=0 "$PY" -u extract_lung1_tumor_patch_embeddings.py --external \
    --cohort "$CHECKED" --cache "$CACHE" --output-root "$ROOT/outputs/radiogenomics_embeddings"
}
fm() {  # model gpu python
  step "fm_$1" env CUDA_VISIBLE_DEVICES="$2" "$3" -u extract_lung1_e1_embeddings.py \
    --model "$1" --source original_ct --cohort "$COHORT" --status-column global_status \
    --clamp-min-hu -1024 --output-root "$ROOT/outputs/radiogenomics_embeddings" --device cuda:0 || true
}
fm_track() {  # three GPU queues; DINOv3 needs the transformers >= 4.56 environment, as for LUNG1
  { fm colipri 3 "$PY"; fm ctclip 3 "$PY"; } &
  { fm tangerine 1 "$PY"; fm radjepa 1 "$PY"; } &
  fm dinov3 2 "$ROOT/.venv-dinov3/bin/python" &
  wait
}
tumour_track &
fm_track &
wait
echo "[$(date '+%F %T')] preparation finished"
touch "$LOGS/ALL_DONE"
