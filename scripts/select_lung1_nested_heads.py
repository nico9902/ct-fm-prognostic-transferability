"""Select the MLP/SQA epoch for each outer fold from the nested-CV inner stage.

Protocol: docs/lung1_sweep_validation_protocol.md. A candidate epoch's score is its mean
validation Uno C at 36 months over inner folds 0-2; the highest mean is selected separately
for each outer fold, method and encoder, ties going to fewer epochs. A nonfinite score
excludes that epoch in that inner fold, and an epoch missing in any inner fold is not eligible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADS = ROOT / "outputs/lung1_nested/heads"
MODELS = {"mlp": ["colipri", "tangerine", "radjepa", "ctclip", "dinov3", "pai", "hosny"],
          "sqa": ["colipri", "tangerine", "radjepa", "ctclip", "dinov3"]}
EPOCHS = {"mlp": 100, "sqa": 40}


def history(method: str, model: str, inner: int, fold: int) -> dict[int, float]:
    payload = json.loads((HEADS / f"inner{inner}" / method / model / "folds" / f"fold_{fold}.json").read_text())
    return {int(h["epoch"]): float(h["val_uno_c_36m"]) for h in payload["validation_history"]
            if math.isfinite(float(h["val_uno_c_36m"]))}


def main() -> None:
    selection: dict = {}
    for method, models in MODELS.items():
        for model in models:
            per_fold = {}
            for fold in range(5):
                inner = [history(method, model, k, fold) for k in range(3)]
                eligible = [e for e in range(1, EPOCHS[method] + 1) if all(e in h for h in inner)]
                means = {e: sum(h[e] for h in inner) / 3 for e in eligible}
                # max over ascending epochs keeps the earliest epoch on ties
                best = max(eligible, key=lambda e: means[e])
                per_fold[str(fold)] = {"epochs": best, "mean_validation_uno_c_36m": means[best],
                                       "inner_validation_uno_c_36m": [h[best] for h in inner]}
            selection.setdefault(method, {})[model] = per_fold
    out = HEADS / "final" / "selected_epochs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2) + "\n")
    for method, models in selection.items():
        for model, folds in models.items():
            print(method, model, {k: v["epochs"] for k, v in folds.items()})


if __name__ == "__main__":
    main()
