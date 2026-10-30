#!/usr/bin/env python3
"""Finish the paired crop E1 runs after all four crop embeddings are complete."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("tangerine", "radjepa", "colipri", "ctclip")
LOGS = ROOT / "logs/lung1_e1_crop"
OUTPUTS = ROOT / "outputs"
PYTHON = ROOT / ".venv/bin/python"
EXPECTED = 416


def alive(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        os.kill(int(pid_path.read_text().strip()), 0)
        return True
    except ProcessLookupError:
        return False


def complete_manifest(model: str) -> bool:
    path = OUTPUTS / f"lung1_e1_embeddings/{model}/thorax_hu/embedding_manifest.csv"
    if not path.exists():
        return False
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return len(rows) == EXPECTED and all(Path(row["embedding_path"]).is_file() for row in rows)


def run_probe(kind: str, components: int) -> None:
    destination = OUTPUTS / f"lung1_e1_frozen_cox_crop_{kind}"
    if (destination / "summary.json").exists():
        print(f"Already complete: {destination}", flush=True)
        return
    command = [str(PYTHON), "-u", str(ROOT / "scripts/run_lung1_frozen_cox_e1.py"),
               "--components", str(components), "--output-dir", str(destination)]
    if components == 0:
        command.extend(("--cox-device", "cuda:0"))
    for model in MODELS:
        manifest = OUTPUTS / f"lung1_e1_embeddings/{model}/thorax_hu/embedding_manifest.csv"
        command.extend(("--embeddings", f"{model}={manifest}"))
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "3"
    with (LOGS / f"crop_{kind}_cox.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(f"Complete: {destination}", flush=True)


def main() -> None:
    while not all(complete_manifest(model) for model in MODELS):
        incomplete = [model for model in MODELS if not complete_manifest(model)]
        for model in incomplete:
            if not alive(LOGS / f"{model}_thorax_hu.pid"):
                raise RuntimeError(f"{model} extraction ended before all {EXPECTED} patients")
        print(f"Waiting for crop embeddings: {', '.join(incomplete)}", flush=True)
        time.sleep(30)
    run_probe("pca32", 32)
    while not (OUTPUTS / "lung1_e1_frozen_cox_original_ridge/summary.json").exists():
        if not alive(LOGS / "original_ridge_cox.pid"):
            raise RuntimeError("Original full-feature ridge Cox ended without a summary")
        print("Waiting for original ridge Cox to release the GPU", flush=True)
        time.sleep(30)
    run_probe("ridge", 0)


if __name__ == "__main__":
    main()
