#!/usr/bin/env python3
"""Compatibility wrapper for the Lightning trainer in src/."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lung_jepa.training.train_latent_predictor import main


if __name__ == "__main__":
    main()
