#!/usr/bin/env python3
"""Generate the frozen TANGERINE comparison report and component sensitivity results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_diagnostic_nonlinear_decodability import metrics

TASKS = ["isolation_suspicious", "nodule_ge_4mm", "emphysema", "fibrosis_or_scarring"]
METRICS = ["auroc", "auprc", "mcc", "accuracy", "f1", "precision", "recall", "specificity", "tn", "fp", "fn", "tp"]


def table(d):
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join(lines)


def formatted(summary, fields):
    rows = []
    for r in summary.itertuples(index=False):
        row = {"Target": r.task, "Probe": r.model}
        for field in fields:
            row[field.upper()] = f"{getattr(r, field+'_mean'):.3f} ± {getattr(r, field+'_std'):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/tangerine_decodability"))
    p.add_argument("--report", type=Path, default=Path("docs/nlst_tangerine_decodability_report.md"))
    a = p.parse_args()
    d = pd.read_csv(a.output_dir / "experiment_scan_manifest.csv", dtype=str)
    original = pd.read_csv(a.root / "experiment_scan_manifest.csv", dtype=str)
    pd.testing.assert_frame_equal(d, original)
    results = pd.read_csv(a.output_dir / "per_seed_results.csv")
    assert len(results) == 40 and not results.duplicated(["task", "model", "seed"]).any()
    assert set(results.task) == set(TASKS) and set(results.model) == {"linear", "mlp"}
    assert results.n_test.eq(597).all()
    summary = pd.read_csv(a.output_dir / "summary_results.csv")
    meta = json.loads((a.output_dir / "metadata.json").read_text())
    enc = json.loads((a.output_dir / "cache/encoder_metadata.json").read_text())
    smoke = json.loads((a.output_dir / "smoke_shapes.json").read_text())
    counts = pd.read_csv(a.output_dir / "overlap_counts.csv")
    baseline = pd.read_csv(a.root / "summary_results.csv")
    baseline = baseline[baseline.representation.ne("sliding_gatedattention")]
    sensitivity = []
    for run in results.itertuples(index=False):
        predictions = pd.read_csv(a.output_dir / "runs" / f"{run.task}_{run.model}_{run.seed}" / "test_predictions.csv", dtype={"patient_id": str, "timepoint": str, "series_uid": str})
        pd.testing.assert_frame_equal(predictions[["patient_id", "timepoint", "series_uid"]].reset_index(drop=True),
                                      d.loc[d.split.eq("test"), ["patient_id", "timepoint", "series_uid"]].reset_index(drop=True))
        for component, prefix in [("ACRIN", "2"), ("LSS", "1")]:
            sub = predictions[predictions.patient_id.str.startswith(prefix)]
            sensitivity.append({"task": run.task, "model": run.model, "seed": run.seed, "component": component,
                                "n_test_scans": len(sub), "n_test_patients": sub.patient_id.nunique(),
                                **metrics(sub.label.to_numpy(), sub.probability.to_numpy())})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(a.output_dir / "component_per_seed_results.csv", index=False)
    agg = sensitivity.groupby(["component", "task", "model"])[METRICS].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.to_csv(a.output_dir / "component_summary_results.csv", index=False)
    rad_root = a.output_dir / "radjepa_reproduced"
    rerun = pd.read_csv(rad_root / "per_seed_results.csv")
    historical = pd.read_csv(a.root / "per_seed_results.csv")
    check = rerun.merge(historical, on=["task", "representation", "seed"], suffixes=("_rerun", "_historical"), validate="one_to_one")
    deviations = {m: float((check[m+'_rerun']-check[m+'_historical']).abs().max()) for m in METRICS}
    (a.output_dir / "radjepa_reproduction_check.json").write_text(json.dumps(deviations, indent=2))
    assert len(check) == 20 and max(deviations.values()) < 1e-8
    rad_sub = []
    for run in rerun.itertuples(index=False):
        pred = pd.read_csv(rad_root / f"{run.task}_{run.seed}_test_predictions.csv", dtype={"patient_id": str})
        for component, prefix in [("ACRIN", "2"), ("LSS", "1")]:
            sub = pred[pred.patient_id.str.startswith(prefix)]
            rad_sub.append({"task": run.task, "representation": run.representation, "seed": run.seed,
                            "component": component, **metrics(sub.label.to_numpy(), sub.probability.to_numpy())})
    rad_sub = pd.DataFrame(rad_sub)
    rad_sub.to_csv(a.output_dir / "radjepa_component_per_seed_results.csv", index=False)
    rad_agg = rad_sub.groupby(["component", "task", "representation"])[METRICS].agg(["mean", "std"]).reset_index()
    rad_agg.columns = ["_".join(c).strip("_") for c in rad_agg.columns]
    rad_agg.to_csv(a.output_dir / "radjepa_component_summary_results.csv", index=False)
    acrin_comparison = []
    for task in TASKS:
        r = rad_agg[rad_agg.component.eq("ACRIN") & rad_agg.task.eq(task)].iloc[0]
        row = {"Target": task, "Rad-JEPA linear (ACRIN)": f"{r.auroc_mean:.3f} ± {r.auroc_std:.3f}"}
        for model in ["linear", "mlp"]:
            s = agg[agg.component.eq("ACRIN") & agg.task.eq(task) & agg.model.eq(model)].iloc[0]
            row[f"TANGERINE {model} (ACRIN)"] = f"{s.auroc_mean:.3f} ± {s.auroc_std:.3f}"
        acrin_comparison.append(row)
    pd.DataFrame(acrin_comparison).to_csv(a.output_dir / "acrin_comparison.csv", index=False)
    # Patient bootstrap for the difference between five-seed mean-probability ensembles.
    # This measures sampling uncertainty separately from optimization-seed variability.
    bootstrap = []
    for task in TASKS:
        base_predictions = [pd.read_csv(rad_root / f"{task}_{seed}_test_predictions.csv", dtype={"patient_id": str}) for seed in range(5)]
        base = base_predictions[0]
        bp = np.stack([p.probability.to_numpy() for p in base_predictions]).mean(0)
        for model in ["linear", "mlp"]:
            tang = [pd.read_csv(a.output_dir / "runs" / f"{task}_{model}_{seed}" / "test_predictions.csv", dtype={"patient_id": str}) for seed in range(5)]
            for pred in tang:
                pd.testing.assert_frame_equal(pred[["patient_id", "timepoint", "series_uid", "label"]], base[["patient_id", "timepoint", "series_uid", "label"]])
            tp = np.stack([p.probability.to_numpy() for p in tang]).mean(0)
            for component in ["all", "ACRIN"]:
                sub = np.ones(len(base), dtype=bool) if component == "all" else base.patient_id.str.startswith("2").to_numpy()
                ids = base.loc[sub, "patient_id"].to_numpy()
                y = base.loc[sub, "label"].to_numpy()
                b, t = bp[sub], tp[sub]
                groups = [np.flatnonzero(ids == pid) for pid in np.unique(ids)]
                rng = np.random.default_rng(42)
                deltas = []
                for _ in range(2000):
                    ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                    if len(np.unique(y[ix])) == 2:
                        deltas.append(roc_auc_score(y[ix], t[ix])-roc_auc_score(y[ix], b[ix]))
                low, high = np.quantile(deltas, [.025, .975])
                bootstrap.append({"task": task, "model": model, "component": component,
                    "radjepa_ensemble_auroc": roc_auc_score(y, b), "tangerine_ensemble_auroc": roc_auc_score(y, t),
                    "delta_auroc": roc_auc_score(y, t)-roc_auc_score(y, b), "ci95_low": low, "ci95_high": high,
                    "bootstrap_samples": len(deltas), "n_patients": len(groups)})
    bootstrap = pd.DataFrame(bootstrap)
    bootstrap.to_csv(a.output_dir / "paired_patient_bootstrap.csv", index=False)
    comparison = pd.read_csv(a.output_dir / "comparison.csv")
    wide = []
    for task in TASKS:
        b = baseline[baseline.task.eq(task)].sort_values("auroc_mean", ascending=False).iloc[0]
        for metric in METRICS:
            # Original baseline summary omits confusion-count aggregates.
            raw = pd.read_csv(a.root / "per_seed_results.csv")
            v = raw.loc[raw.task.eq(task) & raw.representation.eq(b.representation), metric]
            row = {"Target": task, "Metric": metric, "Rad-JEPA best linear": f"{v.mean():.3f} ± {v.std():.3f}"}
            for model in ["linear", "mlp"]:
                s = summary[summary.task.eq(task) & summary.model.eq(model)].iloc[0]
                row[f"TANGERINE {model}"] = f"{s[metric+'_mean']:.3f} ± {s[metric+'_std']:.3f}"
            wide.append(row)
    pd.DataFrame(wide).to_csv(a.output_dir / "comparison_all_metrics.csv", index=False)
    emb = np.load(a.output_dir / "cache/embeddings.npy")
    assert emb.shape == (2984, 2048) and np.isfinite(emb).all()
    qc = {"n_scans": len(d), "n_patients": d.patient_id.nunique(), "embedding_shape": list(emb.shape),
          "finite": True, "constant_features": int((emb.std(axis=0) == 0).sum()),
          "label_prevalence_test": {t: float(pd.to_numeric(d.loc[d.split.eq('test'), t]).mean()) for t in TASKS},
          "identical_cohort_and_test_patients": True, "n_runs": len(results),
          "complete_requested_metric_set": all(m in results for m in METRICS)}
    (a.output_dir / "qc.json").write_text(json.dumps(qc, indent=2))
    shape_rows = pd.DataFrame([{"Tensor": k, "Shape": str(v)} for k, v in smoke.items() if isinstance(v, list)])
    findings = []
    for task in TASKS:
        b = baseline[baseline.task.eq(task)].sort_values("auroc_mean", ascending=False).iloc[0]
        lin = summary[summary.task.eq(task) & summary.model.eq("linear")].iloc[0]
        mlp = summary[summary.task.eq(task) & summary.model.eq("mlp")].iloc[0]
        acr = agg[agg.component.eq("ACRIN") & agg.task.eq(task) & agg.model.eq("linear")].iloc[0]
        findings.append(f"- `{task}`: TANGERINE linear changes AUROC by {lin.auroc_mean-b.auroc_mean:+.3f} relative to the historical best Rad-JEPA linear representation; MLP changes it by {mlp.auroc_mean-lin.auroc_mean:+.3f} relative to TANGERINE linear. ACRIN-subset linear AUROC is {acr.auroc_mean:.3f} ± {acr.auroc_std:.3f}.")
    text = f"""# Frozen TANGERINE versus Rad-JEPA on NLST

## Pretraining overlap — read before interpreting performance

The [TANGERINE paper](https://www.nature.com/articles/s43856-025-01328-1) explicitly identifies NLST-LSS as a major pretraining source. [Supplementary Data 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43856-025-01328-1/MediaObjects/43856_2025_1328_MOESM3_ESM.xlsx) lists 30,760 LSS series from 10,541 patients. This is self-supervised exposure, but it prevents treating potentially exposed test patients as an independent external validation cohort.

Using the [official NLST PID mapping](https://www.cancerimagingarchive.net/wp-content/uploads/NLST_Database_Query_Tool_Data_Dictionary_2012_10_25.pdf), 1xxxxx identifies LSS and 2xxxxx identifies ACRIN. Our cohort contains 664 LSS patients (1,988 scans) and 335 ACRIN patients (996 scans). Ten DICOM PatientName headers from each component were checked and all agreed with the PID mapping. The fixed test split contains 131 potentially exposed LSS patients / 391 scans and 69 ACRIN patients / 206 scans.

{table(counts)}

The released weights and dataset table contain no patient- or series-level pretraining manifest. Exact overlap with these 664 LSS patients cannot be established. ACRIN was described as unseen during pretraining in the paper; its test subset is a useful sensitivity analysis, though absence of individual membership records or cross-dataset duplication checks prevents a definitive patient-level non-overlap guarantee. Subset probes below were trained on the original full training split; they were not retuned on the subsets.

## Scientific answer

This experiment does **not** show substantially stronger full-cohort decoding of screening suspicion or >=4 mm nodule presence from the official frozen TANGERINE whole-volume embedding. Linear AUROCs of 0.542 and 0.528 remain weak; nodule MCC is 0.005 and linear nodule AUPRC does not improve over the selected Rad-JEPA baseline. The secondary MLP lowers mean AUROC for every target, though it modestly raises AUPRC for suspicion and nodules.

The clear positive finding is emphysema: linear AUROC rises from 0.555 to 0.700, AUPRC from 0.437 to 0.612, and MCC from 0.088 to 0.264. Fibrosis/scarring improves more modestly (linear AUROC 0.554 to 0.603). These characterize accessible representation information, not established clinical utility.

ACRIN-only linear suspicion/nodule AUROCs are higher (0.611/0.576), but remain modest and must be read with the patient-bootstrap uncertainty below. In the potentially exposed LSS subset, their linear AUROCs are approximately chance (0.498/0.496). The main negative nodule/suspicion finding applies to this official global embedding and these probe architectures; it does not establish that localized patch representations contain no relevant information.

For the five-seed mean-probability ensembles, the full-cohort emphysema AUROC gain is +0.147 (exploratory paired-patient 95% bootstrap interval +0.075 to +0.219). Full-cohort intervals for suspicion, nodules, and fibrosis/scarring include zero. On ACRIN, the linear suspicion gain is +0.098 (interval +0.005 to +0.188), suggesting a modest subgroup improvement; the ACRIN nodule interval includes zero. These intervals are conditional on this fixed split and fitted probes, and are not adjusted for multiple comparisons.

## Official implementation and frozen representation

Source: [official repository](https://github.com/niccolo246/3D-MAE-MedImaging), commit `{enc['official_commit']}`. We downloaded [Zenodo v1.0](https://zenodo.org/records/18835750), `tangerine.pth` (1,362,743,936 bytes), SHA256 `{enc['checkpoint_sha256']}`.

The encoder is ViT-Large: one input channel, 256³ voxels, 16³ patches, 4,096 patch tokens, embedding width 1,024, 24 transformer blocks, and 16 attention heads. CTs are resampled to 256³ with SimpleITK B-spline interpolation; output spacing is `original_size * original_spacing / 256`, with origin, direction, and physical field of view retained. Resampled HU values are clipped to [-1200,800] and mapped to [0,1] via `(HU+1200)/2000`. Evaluation uses no augmentation or masking.

We instantiate `models_vit.vit_large_patch16_yo(img_size=256, num_classes=0, global_pool=True)`, load the checkpoint's `model`/`model_state` tensor mapping, and follow the official matching-key/shape filter. All {enc['loaded_keys']} encoder tensors loaded; missing keys are only `fc_norm.weight/bias`. As in the official extractor, this new LayerNorm retains its deterministic defaults (weight=1, bias=0, eps=1e-6), and is frozen. Decoder and MAE final normalization tensors are excluded, as in the official global-pooling downstream architecture. Missing encoder tensors cause a hard failure rather than silently retaining random weights.

The official global-pooling path concatenates **CLS with mean-pooled patch tokens** (excluding CLS from the mean), in that order: `[B,2048]`. Crucially, the [official embedding utility](https://github.com/niccolo246/3D-MAE-MedImaging/blob/{enc['official_commit']}/mains_predict/extract_encoder_embeddings.py) defaults to `--embedding_type pre_logits`: **`fc_norm(CLS + mean-patch concatenation)`**, rather than the raw concatenation, CLS-only, or mean-only. We follow this default and cache those normalized 2,048-dimensional features before the classifier. The paper also describes 2,048-dimensional downstream embeddings. Every encoder and normalization parameter is frozen, the model stays in evaluation mode, and all encoder inference uses `torch.inference_mode()` with float32 computation. No encoder fine-tuning was performed. The first 206 raw concatenations cached during extraction were normalized offline using the same fixed LayerNorm, avoiding repeated CT encoding; all subsequent scans were normalized directly after encoder inference.

## Input compatibility and cohort preservation

The prior `[1,D,256,256]` Rad-JEPA arrays are foreground-cropped, resized in-plane, and clipped to [-1000,400] before mapping to [0,1]. They cannot reproduce official TANGERINE preprocessing exactly: intensity clipping and removed field of view are irreversible. The implemented `prepared_approximation` option maps them back with `1400*x-1000`, resamples, and applies the TANGERINE HU scale, but retains those limitations.

The reported primary experiment uses **original HU CTs from the same selected DICOM series** (`source_mode={meta['source_mode']}`), to follow the official TANGERINE pipeline as closely as possible. It checks original depth against each prior QC-approved array. This is an explicit input-source adaptation; it preserves the scan identity and labels while avoiding the irreversible Rad-JEPA intensity preprocessing. Performance differences therefore compare each model with its intended preprocessing and cannot be attributed solely to architecture or pretraining objective.

The original analyzed manifest is reused verbatim: 2,984 scans from 999 patients, with the same eleven preprocessing-QC failures and six missing/ambiguous-linkage exclusions already omitted. Patient splits remain 599/200/200 and scan counts 1,789/598/597 for train/validation/test. No splits were regenerated and no additional scans were dropped. Manifest SHA256: `{meta['cohort_sha256']}`; split SHA256: `{meta['split_sha256']}`. Final checks confirmed identical scan keys, labels, and all 200 test patients.

## Smoke test

{table(shape_rows)}

The smoke forward pass returned finite features and ran before full extraction. Patch and encoder-token shapes were captured with actual forward hooks; both probe architectures were also run on a cached embedding to verify their hidden/logit shapes.

## Probe protocol

For each target: `Linear(2048,1)` first, then `Linear(2048,128) -> ReLU -> Dropout(0.2) -> Linear(128,1)`. Encoder features are reused unchanged. Seeds: 0,1,2,3,4 (40 runs total). StandardScaler is fit on training scans only. BCE-with-logits uses training-only negative/positive weighting; optimizer is AdamW (lr 0.001, weight decay 0.0001); batch size 128; at most 100 epochs; validation-loss early stopping with patience 15 and min_delta 0.00001; deterministic Lightning training. The original Rad-JEPA linear study evaluated weights at the early-stopping epoch without restoring the best checkpoint. We preserve that behavior for both probes. Test probability threshold is 0.5. AUPRC is average precision; specificity is TN/(TN+FP). Standard deviations use sample SD (ddof=1).

## Requested comparison (test AUROC, mean ± SD)

{table(comparison)}

“Rad-JEPA best linear” uses the largest historical test AUROC among Resize32, Sliding-Mean, Sliding-Max, and Sliding-MeanMax for each target. Learned gated attention is excluded because it is not a pure linear probe. This retrospective test-selected baseline is descriptive, not a validation-selected model choice. All models use identical test patients. `comparison_all_metrics.csv` gives the comparison for every requested metric.

## All requested TANGERINE metrics

{table(formatted(summary, ['auroc','auprc','mcc']))}

{table(formatted(summary, ['accuracy','f1','precision','recall','specificity']))}

{table(formatted(summary, ['tn','fp','fn','tp']))}

## ACRIN test-subset sensitivity (69 patients, 206 scans)

{table(pd.DataFrame(acrin_comparison))}

{table(formatted(agg[agg.component.eq('ACRIN')], ['auroc','auprc','mcc']))}

The historical best Rad-JEPA linear probes were reproduced to export keyed predictions: every requested full-test metric matched the original across all 20 runs, within 1e-8. The sensitivity comparison therefore uses identical ACRIN scans and patients and the original full-cohort training protocol. The LSS sensitivity results and all twelve metrics for both components are in `component_summary_results.csv` and `radjepa_component_summary_results.csv`.

## Paired patient bootstrap on ACRIN

{table(bootstrap[bootstrap.component.eq('ACRIN')][['task','model','delta_auroc','ci95_low','ci95_high']].round(3))}

These exploratory percentile 95% intervals use 2,000 paired resamples of test patients with replacement (bootstrap seed 42), retaining all scans for each sampled patient. They compare AUROC of the five-seed mean-probability ensembles, so they differ slightly from the mean per-seed metrics above. They measure test-patient sampling uncertainty, conditional on the fitted probes and fixed split; they do not capture retraining, cohort selection, or unknown pretraining overlap. No correction for multiple comparisons is applied. Full-cohort intervals are also in `paired_patient_bootstrap.csv`.

## Interpretation and limits

{chr(10).join(findings)}

Assess the suspicion and nodule rows as the primary scientific outcomes, alongside AUPRC relative to test prevalences ({qc['label_prevalence_test']}). Improvements in the potentially exposed full-cohort test split characterize accessible frozen information, but cannot establish superiority on wholly unseen NLST patients. The ACRIN subset offers evidence about transfer to a dataset component reported as unseen, with only 69 test patients. Five-seed SD quantifies optimization variability; the separate exploratory patient-bootstrap intervals characterize sampling uncertainty conditional on these fitted models. The selected pilot, read-derived labels, intended preprocessing differences, and unknown exact pretraining membership limit interpretation.

## Artifacts and reproduction

Artifact root: `outputs/tangerine_decodability/`. Cache `cache/scan_embeddings.npz` stores one `embedding` per scan together with `patient_id`, `screening_round`, and `series_uid`; `cache/embeddings.npy` and `cache/embedding_metadata.csv` provide equivalent indexed storage. Each scan is also cached atomically for resumable extraction. `smoke_shapes.json`, `cache/encoder_metadata.json`, `metadata.json`, `qc.json`, and overlap audit files record provenance. `per_seed_results.csv` and `summary_results.csv` contain every requested metric including confusion counts. `runs/` contains probe state dictionaries and keyed test predictions. Training-only scaler parameters are saved in `scaler.npz`.

```bash
.venv/bin/pip install timm==0.9.16 SimpleITK==2.4.1
.venv/bin/python scripts/run_tangerine_decodability.py --device cuda:3 --encode-batch 2 --workers 24
.venv/bin/python scripts/export_radjepa_linear_comparison_predictions.py --gpu 1
.venv/bin/python scripts/summarize_tangerine_decodability.py
```

GPU access in this session requires execution outside the filesystem sandbox. The extractor refuses mixed-cohort, mixed-preprocessing, or mixed-checkpoint cache reuse. Torch {enc['torch']}, official timm 0.9.16, SimpleITK 2.4.1. Model weights are CC-BY-NC 4.0.
"""
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(text)
    (a.output_dir / "report.md").write_text(text)
    print(table(comparison))
    print(f"report={a.report}")


if __name__ == "__main__":
    main()
