#!/usr/bin/env python3
"""E4 token masking without refitting (phase 1.4).

The E4 token-masking probes refit a separate Cox model per condition, so they ask
whether the masked representation still *contains* prognostic information. Here the
Cox model fitted on the ORIGINAL representation (same fold, same selected penalty,
PCA16) is applied unchanged to the masked representation of the held-out patients.
This asks whether the original model *relies* on the tumor tokens.

The refit of the original model must reproduce the saved E4 held-out risks exactly;
the script stops otherwise. Paired intervals resample held-out patients within each
locked fold (repetition zero), shared by all conditions of a model.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from analyze_lung1_e3_frozen_linear import bootstrap_scores, weighted_pair_matrices
from run_lung1_e3_frozen_linear import write_json
from run_lung1_frozen_cox_e1 import COHORT, FOLDS, GRID, ROOT, cohort, fit_cox, fold_metrics, locked_folds, predict, sha256

MODELS = ("colipri", "tangerine", "radjepa", "ctclip", "dinov3")
E4 = ROOT / "outputs/lung1_e4_token_masking"
CONTROLS = ROOT / "outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv"
OUTPUT = ROOT / "outputs/lung1_e4_no_refit"
BOOT_SEED = 20261019


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    full = cohort(COHORT)
    folds = locked_folds(full, FOLDS)
    control = pd.read_csv(CONTROLS, dtype={"patient_id": str})
    usable = set(control.loc[control.sham_dx_voxels.notna(), "patient_id"])
    settings = {
        "full416": (full, "cox_full416", {"keep_gtv": ("gtv_masked",), "global": ("gtv_masked",)}),
        "control403": (full.loc[full.patient_id.isin(usable)].reset_index(drop=True), "cox_control403",
                       {"keep_both": ("gtv_masked", "control_masked"), "global": ("gtv_masked", "control_masked")}),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "protocol.json", {
        "cohort_sha256": sha256(COHORT), "folds_sha256": sha256(FOLDS), "outer_repetition": 0,
        "source_fits": {k: str(E4 / v[1]) for k, v in settings.items()},
        "design": "Cox (PCA16, E4-selected penalty) fitted on original development vectors, applied to masked test vectors",
        "bootstrap": args.bootstrap, "bootstrap_seed": BOOT_SEED})
    summary, differences = [], []
    for setting, (data, source, readouts) in settings.items():
        rng = np.random.default_rng(BOOT_SEED)
        counts, development, split = {}, {}, {}
        for fold in range(5):
            ids = set(folds.loc[(folds.repetition == 0) & (folds.fold == fold), "patient_id"])
            test = data.patient_id.isin(ids).to_numpy()
            split[fold] = test
            sampled = rng.integers(0, int(test.sum()), size=(args.bootstrap, int(test.sum())))
            counts[fold] = np.stack([np.bincount(row, minlength=int(test.sum())) for row in sampled])
            development[fold] = (data.os_time_days.to_numpy(float)[~test], data.os_event.to_numpy(int)[~test])
        for model in MODELS:
            arrays = [np.load(E4 / "embeddings" / model / "vectors" / f"{pid}.npz") for pid in data.patient_id]
            for readout, conditions in readouts.items():
                vectors = {c: np.stack([a[f"{c}__{readout}"] for a in arrays]).astype(np.float32)
                           for c in ("original", *conditions)}
                variants = {"original": [], **{f"{c}_no_refit": [] for c in conditions},
                            **{f"{c}_refit": [] for c in conditions}}
                for fold in range(5):
                    test = split[fold]
                    t_dev, e_dev = development[fold]
                    t_test = data.os_time_days.to_numpy(float)[test]
                    e_test = data.os_event.to_numpy(int)[test]
                    saved = json.loads((E4 / source / "fits" / model / f"{readout}__original" / f"fold_{fold}.json").read_text())
                    fitted = fit_cox(vectors["original"][~test], t_dev, e_dev, saved["alpha"], 16)
                    risk, _ = predict(fitted, vectors["original"][test], GRID)
                    reference = np.array([p["risk"] for p in saved["predictions"]])
                    if [p["patient_id"] for p in saved["predictions"]] != data.patient_id[test].tolist() \
                            or not np.allclose(risk, reference, rtol=1e-6, atol=1e-8):
                        raise ValueError(f"{setting}/{model}/{readout} fold {fold}: original refit does not reproduce E4")
                    for name, x in [("original", vectors["original"]),
                                    *[(f"{c}_no_refit", vectors[c]) for c in conditions]]:
                        r, s = predict(fitted, x[test], GRID)
                        m = fold_metrics(t_dev, e_dev, t_test, e_test, r, s)
                        m["harrell_c_full"] = float(concordance_index(t_test, -r, e_test))
                        variants[name].append((m, r))
                    for c in conditions:  # E4 refit on the masked condition, for comparison
                        refit = json.loads((E4 / source / "fits" / model / f"{readout}__{c}" / f"fold_{fold}.json").read_text())
                        variants[f"{c}_refit"].append((refit["metrics"], np.array([p["risk"] for p in refit["predictions"]])))
                boot = {}
                for name, per_fold in variants.items():
                    draws = []
                    for fold, (_, r) in enumerate(per_fold):
                        test = split[fold]
                        preds = [{"patient_id": p, "os_time_days": t, "os_event": e, "risk": v} for p, t, e, v in
                                 zip(data.patient_id[test], data.os_time_days[test], data.os_event[test], r)]
                        draws.append(bootstrap_scores(*weighted_pair_matrices(preds, *development[fold]), counts[fold]))
                    boot[name] = np.mean(draws, axis=0)
                    original_risk = np.concatenate([r for _, r in variants["original"]])
                    risk_all = np.concatenate([r for _, r in per_fold])
                    summary.append({"setting": setting, "model": model, "readout": readout, "variant": name,
                                    "uno_c_36m": float(np.mean([m["uno_c_36m"] for m, _ in per_fold])),
                                    "ci95_lower": float(np.percentile(boot[name], 2.5)),
                                    "ci95_upper": float(np.percentile(boot[name], 97.5)),
                                    "ibs_3m_36m": float(np.mean([m["ibs_3m_36m"] for m, _ in per_fold])),
                                    "spearman_risk_vs_original": float(pd.Series(risk_all).corr(
                                        pd.Series(original_risk), method="spearman"))})
                point = {s["variant"]: s["uno_c_36m"] for s in summary
                         if (s["setting"], s["model"], s["readout"]) == (setting, model, readout)}
                pairs = [("original", f"{c}_no_refit") for c in conditions]
                if "control_masked" in conditions:
                    pairs.append(("control_masked_no_refit", "gtv_masked_no_refit"))
                for first, second in pairs:
                    d = boot[first] - boot[second]
                    differences.append({"setting": setting, "model": model, "readout": readout,
                                        "contrast": f"{first} - {second}", "difference": point[first] - point[second],
                                        "ci95_lower": float(np.percentile(d, 2.5)),
                                        "ci95_upper": float(np.percentile(d, 97.5))})
                print(f"{setting} {model} {readout}: " + ", ".join(f"{k}={v:.3f}" for k, v in point.items()), flush=True)
    pd.DataFrame(summary).to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(differences).to_csv(OUTPUT / "paired_differences.csv", index=False)
    (OUTPUT / "COMPLETE").write_text("done\n")


if __name__ == "__main__":
    main()
