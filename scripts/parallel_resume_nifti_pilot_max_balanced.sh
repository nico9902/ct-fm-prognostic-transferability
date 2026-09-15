#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG="${TAG:-pilot_1460_balanced}"
JOBS="${JOBS:-4}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/ssd2/domenico/datasets/nlst}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATASET_ROOT}/manifests}"
DICOM_DIR="${DICOM_DIR:-${DATASET_ROOT}/raw/nlst/dicom}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATASET_ROOT}/processed/nlst/${TAG}}"
WORK_DIR="${WORK_DIR:-${PROCESSED_DIR}/parallel_nifti_resume}"
KILL_EXISTING="${KILL_EXISTING:-0}"
FINALIZE_MANIFEST="${FINALIZE_MANIFEST:-1}"

SERIES_MANIFEST="${SERIES_MANIFEST:-${MANIFEST_DIR}/nlst_${TAG}_series.csv}"
NII_ROOT="${NII_ROOT:-${PROCESSED_DIR}/nii}"
FINAL_NII_MANIFEST="${FINAL_NII_MANIFEST:-${PROCESSED_DIR}/nii_manifest.csv}"
MISSING_MANIFEST="${WORK_DIR}/missing_series.csv"
SHARD_DIR="${WORK_DIR}/shards"
LOG_DIR="${WORK_DIR}/logs"

cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}" "${SHARD_DIR}" "${LOG_DIR}" "${NII_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${SERIES_MANIFEST}" ]]; then
  echo "Missing series manifest: ${SERIES_MANIFEST}" >&2
  exit 1
fi

echo "Parallel NIfTI resume"
echo "TAG=${TAG}"
echo "JOBS=${JOBS}"
echo "SERIES_MANIFEST=${SERIES_MANIFEST}"
echo "NII_ROOT=${NII_ROOT}"
echo "WORK_DIR=${WORK_DIR}"

if [[ "${KILL_EXISTING}" == "1" ]]; then
  PIDS="$(pgrep -f "convert_nlst_dicom_to_nifti.py.*${TAG}" || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "Stopping existing conversion processes: ${PIDS}"
    kill ${PIDS} || true
    sleep 5
  fi
fi

if pgrep -af "convert_nlst_dicom_to_nifti.py.*${TAG}" >/dev/null; then
  echo "A conversion process for ${TAG} is already running:" >&2
  pgrep -af "convert_nlst_dicom_to_nifti.py.*${TAG}" >&2
  echo "Rerun with KILL_EXISTING=1 to stop it before starting parallel resume." >&2
  exit 1
fi

"${PYTHON}" - <<PY
from pathlib import Path
import math
import pandas as pd

series_manifest = Path("${SERIES_MANIFEST}")
nii_root = Path("${NII_ROOT}")
missing_manifest = Path("${MISSING_MANIFEST}")
shard_dir = Path("${SHARD_DIR}")
jobs = int("${JOBS}")

df = pd.read_csv(series_manifest, dtype=str)
missing_rows = []
for row in df.itertuples(index=False):
    out_path = nii_root / row.patient_id / f"{row.patient_id}_{row.timepoint}_{row.series_uid}.nii.gz"
    if not out_path.exists():
        missing_rows.append(row._asdict())

missing = pd.DataFrame(missing_rows, columns=df.columns)
missing.to_csv(missing_manifest, index=False)

for old in shard_dir.glob("missing_*.csv"):
    old.unlink()

if missing.empty:
    print("missing_series=0")
else:
    shard_size = math.ceil(len(missing) / jobs)
    for shard_idx, start in enumerate(range(0, len(missing), shard_size)):
        shard = missing.iloc[start:start + shard_size]
        shard.to_csv(shard_dir / f"missing_{shard_idx:03d}.csv", index=False)
    print(f"missing_series={len(missing)}")
    print(f"shards={len(list(shard_dir.glob('missing_*.csv')))}")
PY

MISSING_COUNT="$("${PYTHON}" - <<PY
import pandas as pd
from pathlib import Path
p = Path("${MISSING_MANIFEST}")
print(0 if not p.exists() else len(pd.read_csv(p, dtype=str)))
PY
)"

if [[ "${MISSING_COUNT}" == "0" ]]; then
  echo "No missing NIfTI files. Skipping parallel conversion."
else
  echo "Starting parallel conversion for ${MISSING_COUNT} missing series..."
  pids=()
  for shard in "${SHARD_DIR}"/missing_*.csv; do
    shard_name="$(basename "${shard}" .csv)"
    (
      "${PYTHON}" scripts/convert_nlst_dicom_to_nifti.py \
        --manifest "${shard}" \
        --dicom-root "${DICOM_DIR}" \
        --output-root "${NII_ROOT}" \
        --output-manifest "${WORK_DIR}/nii_manifest_${shard_name}.csv" \
        --on-error skip
    ) > "${LOG_DIR}/${shard_name}.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one shard failed. Check logs in ${LOG_DIR}" >&2
    exit 1
  fi
fi

if [[ "${FINALIZE_MANIFEST}" == "1" ]]; then
  echo "Finalizing full NIfTI manifest..."
  "${PYTHON}" scripts/convert_nlst_dicom_to_nifti.py \
    --manifest "${SERIES_MANIFEST}" \
    --dicom-root "${DICOM_DIR}" \
    --output-root "${NII_ROOT}" \
    --output-manifest "${FINAL_NII_MANIFEST}" \
    --on-error skip
fi

echo "Parallel NIfTI resume completed."
echo "Final manifest: ${FINAL_NII_MANIFEST}"
