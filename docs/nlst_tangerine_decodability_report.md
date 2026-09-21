# Frozen TANGERINE versus Rad-JEPA on NLST

## Pretraining overlap — read before interpreting performance

The [TANGERINE paper](https://www.nature.com/articles/s43856-025-01328-1) explicitly identifies NLST-LSS as a major pretraining source. [Supplementary Data 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43856-025-01328-1/MediaObjects/43856_2025_1328_MOESM3_ESM.xlsx) lists 30,760 LSS series from 10,541 patients. This is self-supervised exposure, but it prevents treating potentially exposed test patients as an independent external validation cohort.

Using the [official NLST PID mapping](https://www.cancerimagingarchive.net/wp-content/uploads/NLST_Database_Query_Tool_Data_Dictionary_2012_10_25.pdf), 1xxxxx identifies LSS and 2xxxxx identifies ACRIN. Our cohort contains 664 LSS patients (1,988 scans) and 335 ACRIN patients (996 scans). Ten DICOM PatientName headers from each component were checked and all agreed with the PID mapping. The fixed test split contains 131 potentially exposed LSS patients / 391 scans and 69 ACRIN patients / 206 scans.

| trial_component | split | scans | patients |
| --- | --- | --- | --- |
| ACRIN | test | 206 | 69 |
| ACRIN | train | 609 | 205 |
| ACRIN | val | 181 | 61 |
| LSS | test | 391 | 131 |
| LSS | train | 1180 | 394 |
| LSS | val | 417 | 139 |

The released weights and dataset table contain no patient- or series-level pretraining manifest. Exact overlap with these 664 LSS patients cannot be established. ACRIN was described as unseen during pretraining in the paper; its test subset is a useful sensitivity analysis, though absence of individual membership records or cross-dataset duplication checks prevents a definitive patient-level non-overlap guarantee. Subset probes below were trained on the original full training split; they were not retuned on the subsets.

## Scientific answer

This experiment does **not** show substantially stronger full-cohort decoding of screening suspicion or >=4 mm nodule presence from the official frozen TANGERINE whole-volume embedding. Linear AUROCs of 0.542 and 0.528 remain weak; nodule MCC is 0.005 and linear nodule AUPRC does not improve over the selected Rad-JEPA baseline. The secondary MLP lowers mean AUROC for every target, though it modestly raises AUPRC for suspicion and nodules.

The clear positive finding is emphysema: linear AUROC rises from 0.555 to 0.700, AUPRC from 0.437 to 0.612, and MCC from 0.088 to 0.264. Fibrosis/scarring improves more modestly (linear AUROC 0.554 to 0.603). These characterize accessible representation information, not established clinical utility.

ACRIN-only linear suspicion/nodule AUROCs are higher (0.611/0.576), but remain modest and must be read with the patient-bootstrap uncertainty below. In the potentially exposed LSS subset, their linear AUROCs are approximately chance (0.498/0.496). The main negative nodule/suspicion finding applies to this official global embedding and these probe architectures; it does not establish that localized patch representations contain no relevant information.

For the five-seed mean-probability ensembles, the full-cohort emphysema AUROC gain is +0.147 (exploratory paired-patient 95% bootstrap interval +0.075 to +0.219). Full-cohort intervals for suspicion, nodules, and fibrosis/scarring include zero. On ACRIN, the linear suspicion gain is +0.098 (interval +0.005 to +0.188), suggesting a modest subgroup improvement; the ACRIN nodule interval includes zero. These intervals are conditional on this fixed split and fitted probes, and are not adjusted for multiple comparisons.

## Official implementation and frozen representation

Source: [official repository](https://github.com/niccolo246/3D-MAE-MedImaging), commit `5bc575f593197097e4d68503bb94bb1c9bb4f1d3`. We downloaded [Zenodo v1.0](https://zenodo.org/records/18835750), `tangerine.pth` (1,362,743,936 bytes), SHA256 `ebd9a60eb8cce3fbc05012dc89a0b075750e166ce4b432d1c75a0e1839702c6c`.

The encoder is ViT-Large: one input channel, 256³ voxels, 16³ patches, 4,096 patch tokens, embedding width 1,024, 24 transformer blocks, and 16 attention heads. CTs are resampled to 256³ with SimpleITK B-spline interpolation; output spacing is `original_size * original_spacing / 256`, with origin, direction, and physical field of view retained. Resampled HU values are clipped to [-1200,800] and mapped to [0,1] via `(HU+1200)/2000`. Evaluation uses no augmentation or masking.

We instantiate `models_vit.vit_large_patch16_yo(img_size=256, num_classes=0, global_pool=True)`, load the checkpoint's `model`/`model_state` tensor mapping, and follow the official matching-key/shape filter. All 292 encoder tensors loaded; missing keys are only `fc_norm.weight/bias`. As in the official extractor, this new LayerNorm retains its deterministic defaults (weight=1, bias=0, eps=1e-6), and is frozen. Decoder and MAE final normalization tensors are excluded, as in the official global-pooling downstream architecture. Missing encoder tensors cause a hard failure rather than silently retaining random weights.

The official global-pooling path concatenates **CLS with mean-pooled patch tokens** (excluding CLS from the mean), in that order: `[B,2048]`. Crucially, the [official embedding utility](https://github.com/niccolo246/3D-MAE-MedImaging/blob/5bc575f593197097e4d68503bb94bb1c9bb4f1d3/mains_predict/extract_encoder_embeddings.py) defaults to `--embedding_type pre_logits`: **`fc_norm(CLS + mean-patch concatenation)`**, rather than the raw concatenation, CLS-only, or mean-only. We follow this default and cache those normalized 2,048-dimensional features before the classifier. The paper also describes 2,048-dimensional downstream embeddings. Every encoder and normalization parameter is frozen, the model stays in evaluation mode, and all encoder inference uses `torch.inference_mode()` with float32 computation. No encoder fine-tuning was performed. The first 206 raw concatenations cached during extraction were normalized offline using the same fixed LayerNorm, avoiding repeated CT encoding; all subsequent scans were normalized directly after encoder inference.

## Input compatibility and cohort preservation

The prior `[1,D,256,256]` Rad-JEPA arrays are foreground-cropped, resized in-plane, and clipped to [-1000,400] before mapping to [0,1]. They cannot reproduce official TANGERINE preprocessing exactly: intensity clipping and removed field of view are irreversible. The implemented `prepared_approximation` option maps them back with `1400*x-1000`, resamples, and applies the TANGERINE HU scale, but retains those limitations.

The reported primary experiment uses **original HU CTs from the same selected DICOM series** (`source_mode=original_hu`), to follow the official TANGERINE pipeline as closely as possible. It checks original depth against each prior QC-approved array. This is an explicit input-source adaptation; it preserves the scan identity and labels while avoiding the irreversible Rad-JEPA intensity preprocessing. Performance differences therefore compare each model with its intended preprocessing and cannot be attributed solely to architecture or pretraining objective.

The original analyzed manifest is reused verbatim: 2,984 scans from 999 patients, with the same eleven preprocessing-QC failures and six missing/ambiguous-linkage exclusions already omitted. Patient splits remain 599/200/200 and scan counts 1,789/598/597 for train/validation/test. No splits were regenerated and no additional scans were dropped. Manifest SHA256: `1678cd22b57450fa46f89df0bdb27d60af38c7a9083b29b3563473d099b5fb59`; split SHA256: `3392c5e0221bf346e1ca7a928f0f8a374db6f6a81639370d55b3e124c3e8d0f0`. Final checks confirmed identical scan keys, labels, and all 200 test patients.

## Smoke test

| Tensor | Shape |
| --- | --- |
| prepared_input | [1, 124, 256, 256] |
| preprocessed_input | [1, 256, 256, 256] |
| batched_input | [1, 1, 256, 256, 256] |
| patch_grid | [16, 16, 16] |
| pos_embed | [1, 4097, 1024] |
| patch_tokens | [1, 4096, 1024] |
| encoder_tokens | [1, 4097, 1024] |
| cls | [1, 1024] |
| mean_patch | [1, 1024] |
| whole_ct_embedding | [1, 2048] |
| linear_logits | [1, 1] |
| mlp_hidden | [1, 128] |
| mlp_logits | [1, 1] |

The smoke forward pass returned finite features and ran before full extraction. Patch and encoder-token shapes were captured with actual forward hooks; both probe architectures were also run on a cached embedding to verify their hidden/logit shapes.

## Probe protocol

For each target: `Linear(2048,1)` first, then `Linear(2048,128) -> ReLU -> Dropout(0.2) -> Linear(128,1)`. Encoder features are reused unchanged. Seeds: 0,1,2,3,4 (40 runs total). StandardScaler is fit on training scans only. BCE-with-logits uses training-only negative/positive weighting; optimizer is AdamW (lr 0.001, weight decay 0.0001); batch size 128; at most 100 epochs; validation-loss early stopping with patience 15 and min_delta 0.00001; deterministic Lightning training. The original Rad-JEPA linear study evaluated weights at the early-stopping epoch without restoring the best checkpoint. We preserve that behavior for both probes. Test probability threshold is 0.5. AUPRC is average precision; specificity is TN/(TN+FP). Standard deviations use sample SD (ddof=1).

## Requested comparison (test AUROC, mean ± SD)

| Target | Rad-JEPA representation | Rad-JEPA best linear | TANGERINE linear | TANGERINE mlp |
| --- | --- | --- | --- | --- |
| isolation_suspicious | sliding_mean | 0.507 ± 0.007 | 0.542 ± 0.013 | 0.534 ± 0.010 |
| nodule_ge_4mm | sliding_mean | 0.504 ± 0.006 | 0.528 ± 0.014 | 0.526 ± 0.019 |
| emphysema | sliding_max | 0.555 ± 0.002 | 0.700 ± 0.011 | 0.688 ± 0.007 |
| fibrosis_or_scarring | sliding_mean | 0.554 ± 0.005 | 0.603 ± 0.018 | 0.588 ± 0.024 |

“Rad-JEPA best linear” uses the largest historical test AUROC among Resize32, Sliding-Mean, Sliding-Max, and Sliding-MeanMax for each target. Learned gated attention is excluded because it is not a pure linear probe. This retrospective test-selected baseline is descriptive, not a validation-selected model choice. All models use identical test patients. `comparison_all_metrics.csv` gives the comparison for every requested metric.

## All requested TANGERINE metrics

| Target | Probe | AUROC | AUPRC | MCC |
| --- | --- | --- | --- | --- |
| emphysema | linear | 0.700 ± 0.011 | 0.612 ± 0.009 | 0.264 ± 0.025 |
| emphysema | mlp | 0.688 ± 0.007 | 0.592 ± 0.010 | 0.262 ± 0.010 |
| fibrosis_or_scarring | linear | 0.603 ± 0.018 | 0.406 ± 0.023 | 0.132 ± 0.018 |
| fibrosis_or_scarring | mlp | 0.588 ± 0.024 | 0.392 ± 0.032 | 0.111 ± 0.043 |
| isolation_suspicious | linear | 0.542 ± 0.013 | 0.375 ± 0.009 | 0.034 ± 0.014 |
| isolation_suspicious | mlp | 0.534 ± 0.010 | 0.391 ± 0.006 | 0.045 ± 0.038 |
| nodule_ge_4mm | linear | 0.528 ± 0.014 | 0.361 ± 0.009 | 0.005 ± 0.024 |
| nodule_ge_4mm | mlp | 0.526 ± 0.019 | 0.379 ± 0.016 | 0.018 ± 0.037 |

| Target | Probe | ACCURACY | F1 | PRECISION | RECALL | SPECIFICITY |
| --- | --- | --- | --- | --- | --- | --- |
| emphysema | linear | 0.642 ± 0.014 | 0.566 ± 0.012 | 0.537 ± 0.017 | 0.599 ± 0.011 | 0.669 ± 0.021 |
| emphysema | mlp | 0.646 ± 0.004 | 0.554 ± 0.018 | 0.545 ± 0.008 | 0.566 ± 0.044 | 0.697 ± 0.032 |
| fibrosis_or_scarring | linear | 0.582 ± 0.013 | 0.440 ± 0.010 | 0.369 ± 0.011 | 0.546 ± 0.019 | 0.597 ± 0.022 |
| fibrosis_or_scarring | mlp | 0.630 ± 0.031 | 0.370 ± 0.022 | 0.384 ± 0.038 | 0.360 ± 0.034 | 0.747 ± 0.053 |
| isolation_suspicious | linear | 0.521 ± 0.006 | 0.426 ± 0.010 | 0.367 ± 0.007 | 0.507 ± 0.014 | 0.529 ± 0.004 |
| isolation_suspicious | mlp | 0.542 ± 0.024 | 0.412 ± 0.022 | 0.375 ± 0.021 | 0.458 ± 0.036 | 0.588 ± 0.043 |
| nodule_ge_4mm | linear | 0.506 ± 0.011 | 0.408 ± 0.014 | 0.348 ± 0.011 | 0.493 ± 0.019 | 0.513 ± 0.009 |
| nodule_ge_4mm | mlp | 0.533 ± 0.027 | 0.389 ± 0.023 | 0.356 ± 0.022 | 0.431 ± 0.044 | 0.587 ± 0.055 |

| Target | Probe | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| emphysema | linear | 243.400 ± 7.570 | 120.600 ± 7.570 | 93.400 ± 2.510 | 139.600 ± 2.510 |
| emphysema | mlp | 253.800 ± 11.713 | 110.200 ± 11.713 | 101.200 ± 10.281 | 131.800 ± 10.281 |
| fibrosis_or_scarring | linear | 249.000 ± 9.327 | 168.000 ± 9.327 | 81.800 ± 3.421 | 98.200 ± 3.421 |
| fibrosis_or_scarring | mlp | 311.600 ± 22.199 | 105.400 ± 22.199 | 115.200 ± 6.140 | 64.800 ± 6.140 |
| isolation_suspicious | linear | 205.200 ± 1.483 | 182.800 ± 1.483 | 103.000 ± 3.000 | 106.000 ± 3.000 |
| isolation_suspicious | mlp | 228.000 ± 16.492 | 160.000 ± 16.492 | 113.200 ± 7.463 | 95.800 ± 7.463 |
| nodule_ge_4mm | linear | 200.400 ± 3.362 | 190.600 ± 3.362 | 104.400 ± 3.912 | 101.600 ± 3.912 |
| nodule_ge_4mm | mlp | 229.600 ± 21.408 | 161.400 ± 21.408 | 117.200 ± 9.094 | 88.800 ± 9.094 |

## ACRIN test-subset sensitivity (69 patients, 206 scans)

| Target | Rad-JEPA linear (ACRIN) | TANGERINE linear (ACRIN) | TANGERINE mlp (ACRIN) |
| --- | --- | --- | --- |
| isolation_suspicious | 0.514 ± 0.010 | 0.611 ± 0.025 | 0.569 ± 0.023 |
| nodule_ge_4mm | 0.504 ± 0.012 | 0.576 ± 0.027 | 0.537 ± 0.051 |
| emphysema | 0.592 ± 0.011 | 0.682 ± 0.019 | 0.680 ± 0.018 |
| fibrosis_or_scarring | 0.463 ± 0.007 | 0.551 ± 0.045 | 0.554 ± 0.055 |

| Target | Probe | AUROC | AUPRC | MCC |
| --- | --- | --- | --- | --- |
| emphysema | linear | 0.682 ± 0.019 | 0.677 ± 0.010 | 0.217 ± 0.034 |
| emphysema | mlp | 0.680 ± 0.018 | 0.659 ± 0.018 | 0.231 ± 0.025 |
| fibrosis_or_scarring | linear | 0.551 ± 0.045 | 0.386 ± 0.036 | 0.065 ± 0.046 |
| fibrosis_or_scarring | mlp | 0.554 ± 0.055 | 0.365 ± 0.043 | 0.033 ± 0.071 |
| isolation_suspicious | linear | 0.611 ± 0.025 | 0.425 ± 0.008 | 0.200 ± 0.069 |
| isolation_suspicious | mlp | 0.569 ± 0.023 | 0.438 ± 0.016 | 0.106 ± 0.048 |
| nodule_ge_4mm | linear | 0.576 ± 0.027 | 0.382 ± 0.010 | 0.131 ± 0.079 |
| nodule_ge_4mm | mlp | 0.537 ± 0.051 | 0.394 ± 0.032 | 0.050 ± 0.089 |

The historical best Rad-JEPA linear probes were reproduced to export keyed predictions: every requested full-test metric matched the original across all 20 runs, within 1e-8. The sensitivity comparison therefore uses identical ACRIN scans and patients and the original full-cohort training protocol. The LSS sensitivity results and all twelve metrics for both components are in `component_summary_results.csv` and `radjepa_component_summary_results.csv`.

## Paired patient bootstrap on ACRIN

| task | model | delta_auroc | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| isolation_suspicious | linear | 0.098 | 0.005 | 0.188 |
| isolation_suspicious | mlp | 0.06 | -0.047 | 0.166 |
| nodule_ge_4mm | linear | 0.069 | -0.027 | 0.165 |
| nodule_ge_4mm | mlp | 0.034 | -0.08 | 0.139 |
| emphysema | linear | 0.092 | -0.035 | 0.217 |
| emphysema | mlp | 0.087 | -0.025 | 0.192 |
| fibrosis_or_scarring | linear | 0.089 | -0.04 | 0.205 |
| fibrosis_or_scarring | mlp | 0.095 | -0.053 | 0.227 |

These exploratory percentile 95% intervals use 2,000 paired resamples of test patients with replacement (bootstrap seed 42), retaining all scans for each sampled patient. They compare AUROC of the five-seed mean-probability ensembles, so they differ slightly from the mean per-seed metrics above. They measure test-patient sampling uncertainty, conditional on the fitted probes and fixed split; they do not capture retraining, cohort selection, or unknown pretraining overlap. No correction for multiple comparisons is applied. Full-cohort intervals are also in `paired_patient_bootstrap.csv`.

## Interpretation and limits

- `isolation_suspicious`: TANGERINE linear changes AUROC by +0.035 relative to the historical best Rad-JEPA linear representation; MLP changes it by -0.008 relative to TANGERINE linear. ACRIN-subset linear AUROC is 0.611 ± 0.025.
- `nodule_ge_4mm`: TANGERINE linear changes AUROC by +0.024 relative to the historical best Rad-JEPA linear representation; MLP changes it by -0.001 relative to TANGERINE linear. ACRIN-subset linear AUROC is 0.576 ± 0.027.
- `emphysema`: TANGERINE linear changes AUROC by +0.145 relative to the historical best Rad-JEPA linear representation; MLP changes it by -0.013 relative to TANGERINE linear. ACRIN-subset linear AUROC is 0.682 ± 0.019.
- `fibrosis_or_scarring`: TANGERINE linear changes AUROC by +0.049 relative to the historical best Rad-JEPA linear representation; MLP changes it by -0.015 relative to TANGERINE linear. ACRIN-subset linear AUROC is 0.551 ± 0.045.

Assess the suspicion and nodule rows as the primary scientific outcomes, alongside AUPRC relative to test prevalences ({'isolation_suspicious': 0.3500837520938023, 'nodule_ge_4mm': 0.34505862646566166, 'emphysema': 0.39028475711892796, 'fibrosis_or_scarring': 0.3015075376884422}). Improvements in the potentially exposed full-cohort test split characterize accessible frozen information, but cannot establish superiority on wholly unseen NLST patients. The ACRIN subset offers evidence about transfer to a dataset component reported as unseen, with only 69 test patients. Five-seed SD quantifies optimization variability; the separate exploratory patient-bootstrap intervals characterize sampling uncertainty conditional on these fitted models. The selected pilot, read-derived labels, intended preprocessing differences, and unknown exact pretraining membership limit interpretation.

## Artifacts and reproduction

Artifact root: `outputs/tangerine_decodability/`. Cache `cache/scan_embeddings.npz` stores one `embedding` per scan together with `patient_id`, `screening_round`, and `series_uid`; `cache/embeddings.npy` and `cache/embedding_metadata.csv` provide equivalent indexed storage. Each scan is also cached atomically for resumable extraction. `smoke_shapes.json`, `cache/encoder_metadata.json`, `metadata.json`, `qc.json`, and overlap audit files record provenance. `per_seed_results.csv` and `summary_results.csv` contain every requested metric including confusion counts. `runs/` contains probe state dictionaries and keyed test predictions. Training-only scaler parameters are saved in `scaler.npz`.

```bash
.venv/bin/pip install timm==0.9.16 SimpleITK==2.4.1
.venv/bin/python scripts/run_tangerine_decodability.py --device cuda:3 --encode-batch 2 --workers 24
.venv/bin/python scripts/export_radjepa_linear_comparison_predictions.py --gpu 1
.venv/bin/python scripts/summarize_tangerine_decodability.py
```

GPU access in this session requires execution outside the filesystem sandbox. The extractor refuses mixed-cohort, mixed-preprocessing, or mixed-checkpoint cache reuse. Torch 2.5.1+cu121, official timm 0.9.16, SimpleITK 2.4.1. Model weights are CC-BY-NC 4.0.
