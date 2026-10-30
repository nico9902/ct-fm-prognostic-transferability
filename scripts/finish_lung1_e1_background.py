#!/usr/bin/env python3
"""Wait for all four frozen embedding jobs, then run the locked E1 Cox audit."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs/lung1_e1"
EMBEDDINGS = ROOT / "outputs/lung1_e1_embeddings"
MODELS = ("tangerine", "radjepa", "colipri", "ctclip")


def worker_alive(pid: int) -> bool:
    status = Path(f"/proc/{pid}/stat")
    if not status.exists():
        return False
    return status.read_text().split(") ", 1)[1][0] != "Z"


def finished(model: str, cohort: pd.DataFrame) -> bool:
    manifest = EMBEDDINGS / model / "original_ct/embedding_manifest.csv"
    log = LOGS / f"{model}_original_ct.log"
    if manifest.is_file() and log.is_file() and "Complete:" in log.read_text()[-2048:]:
        rows = pd.read_csv(manifest, dtype={"patient_id": str, "ct_series_uid": str})
        if len(rows) != len(cohort) or rows.patient_id.duplicated().any():
            raise RuntimeError(f"{model}: completion log but incomplete manifest")
        expected = cohort.set_index("patient_id").ct_series_uid
        actual = rows.set_index("patient_id").ct_series_uid
        if not expected.index.difference(actual.index).empty or not actual.index.difference(expected.index).empty:
            raise RuntimeError(f"{model}: manifest patient IDs differ")
        if not expected.sort_index().equals(actual.sort_index()):
            raise RuntimeError(f"{model}: manifest CT series UIDs differ")
        return True
    pid = int((LOGS / f"{model}_original_ct.pid").read_text().strip())
    if not worker_alive(pid):
        raise RuntimeError(f"{model}: worker {pid} exited before completion; see {log}")
    return False


def main() -> None:
    cohort = pd.read_csv(ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv",
                         dtype={"patient_id": str, "ct_series_uid": str})
    cohort = cohort.loc[cohort.region_status.eq("ready")]
    if len(cohort) != 416:
        raise RuntimeError(f"Expected the locked 416-patient cohort, found {len(cohort)}")
    print("Waiting for four frozen embedding manifests", flush=True)
    while True:
        pending = [model for model in MODELS if not finished(model, cohort)]
        if not pending:
            break
        print("Pending: " + ", ".join(pending), flush=True)
        time.sleep(30)
    command = [sys.executable, str(ROOT / "scripts/run_lung1_frozen_cox_e1.py"),
               "--output-dir", str(ROOT / "outputs/lung1_e1_frozen_cox_four_models")]
    for model in MODELS:
        command.extend(("--embeddings", f"{model}={EMBEDDINGS / model / 'original_ct/embedding_manifest.csv'}"))
    print("All embeddings complete; starting nested Cox audit", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    print("E1 complete: outputs/lung1_e1_frozen_cox_four_models/summary.json", flush=True)


if __name__ == "__main__":
    main()
