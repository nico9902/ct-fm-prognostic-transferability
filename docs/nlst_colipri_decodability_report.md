# Frozen COLIPRI on the fixed NLST cohort

Execution is in progress. No COLIPRI outcome is established until all scan caches and five-seed probes finish.

## Checkpoint and official contract

The [public Microsoft release](https://huggingface.co/microsoft/colipri) identifies its single `model.safetensors` checkpoint as COLIPRI-CRM. Revision `6f2d04e59c8a2ade62f62f0bd376fd757c966983` contains no separate CM checkpoint. CM is a desirable classification ablation in the paper, but cannot be evaluated from the verified release. We use CRM and never relabel it CM.

Official package: colipri 0.1.3. Its processor reorients original HU CTs to RAS, linearly resamples to 2 mm isotropic (antialias disabled), clips to [-1000,1000], scales to [-1,1], and center crops/pads to 192³ with minimum-intensity padding. The paper's earlier 160³ setting differs from the released default; this experiment follows the release. Dense extraction is `model.encode_image(batch, pool=False, project=False, normalize=False)`: [B,864,24,24,24], flattened in x/y/z order to [B,13824,864], with 8³ voxel patches and no CLS/register tokens.

Mean and max pool the raw encoder tokens to 864 dimensions. Attention reuses the frozen official 864→768 Conv3d projector and `AttentionPool1D` (12 heads, query equal to token mean), giving 768 dimensions. The smoke test checks equality to the official `encode_image(pool=True, project=True, normalize=False)` path. This attention arm measures the pretrained report-alignment pool; it is not a new task-trained aggregation head. Feature-space differences are explicit and should be considered when comparing pooling arms.

The [paper, Appendix C.2 and B.1.2](https://arxiv.org/html/2510.15042v2) explores mean, max, learned-query attention, mean-query attention, and four-query attention; it trains new aggregation for classification, with validation AUPRC selection. Max pooling performed well for probing. Our primary protocol instead freezes every pretrained module and trains only `Linear(D,1)`, matching the requested fixed-embedding comparison. A learned pooling experiment would be a separate, nonlinear probe.

## Cohort and interpretation

Reuse the original diagnostic manifest and patient split without regeneration: 2,984 scans, 999 patients; 1,789/598/597 scans and 599/200/200 patients in train/val/test. Existing preprocessing and linkage exclusions remain in effect. CTs come from original DICOM HU from exactly the prior QC-approved series, using the same localizer removal, dominant-shape filtering, sorting and duplicate-position removal. Native depth must match the QC-approved Rad-JEPA array. Any extraction failure stops the experiment; no additional scans are silently removed.

COLIPRI CM/CRM used NLST images through its vision-only MAE objective. This measures **clinical decodability of pretrained representations**, not fully independent external generalization. Exact patient membership in pretraining is unavailable. Better screening suspicion/nodule performance would support accessible information in these pretrained features, not independent validation or demonstrated localization. The default center crop can remove peripheral findings; negative outcomes cannot establish absence of all focal information. A higher-resolution sliding-window ablation remains a separate experiment.

## Probe and cache protocol

Targets: isolation_suspicious, nodule_ge_4mm, emphysema, fibrosis_or_scarring. Seeds 0–4; training-only StandardScaler, training-only BCE class weights, AdamW lr .001 and weight decay .0001, batch 128, up to 100 epochs, validation-loss early stopping patience 15/min_delta 1e-5. Match the historical early-stopping-epoch evaluation without restoring a best checkpoint. Threshold .5; AUPRC is average precision. Report sample SD (ddof=1) for AUROC, AUPRC, MCC, accuracy, F1, precision, recall, specificity, TN/FP/FN/TP. Optional `--mlp` adds Linear(D,128), ReLU, Dropout(.2), Linear(128,1).

Dense grids are cached atomically in float32 (~48 MB/scan, ~143 GB total); pooled embeddings are cached atomically and consolidated with keyed metadata. Metadata binds caches to checkpoint SHA256, original cohort/split hashes, package versions and pooling/preprocessing settings. Resuming with changed provenance is refused. Each pooling directory stores scalers, probe weights, keyed test probabilities and per-seed/summary metrics. Comparison CSVs for every metric include historical best Rad-JEPA pure linear representation (retrospectively test-AUROC selected, descriptive) and TANGERINE linear on the identical cohort.

## Reproduction

```bash
.venv/bin/pip install colipri==0.1.3
.venv/bin/pip install timm==1.0.22 --no-deps --target external/colipri/deps
.venv/bin/python scripts/run_colipri_decodability.py --smoke-only --device cuda:3
.venv/bin/python scripts/run_colipri_decodability.py --device cuda:3
```

The isolated timm install preserves TANGERINE's timm 0.9.16. Checkpoint must be downloaded at the revision above to `external/colipri/model.safetensors`. Artifacts: `outputs/colipri_decodability/`; smoke shapes in `smoke_shapes.json`, immutable provenance in `metadata.json`, all requested metrics in `summary_results.csv` and `comparison_<metric>.csv` after completion. GPU inference requires sandbox escalation in this session.

## Completed smoke test

The smoke test passed before full extraction; all dense and pooled features were finite, and reused attention matched official pooled inference. Shapes use TorchIO channel/x/y/z order.

| Tensor | Actual shape |
| --- | --- |
| original_ct_shape | [1, 512, 512, 124] |
| resampled_ct_shape | [1, 144, 144, 124] |
| colipri_input_shape | [1, 1, 192, 192, 192] |
| dense_grid_shape | [1, 864, 24, 24, 24] |
| dense_token_shape | [1, 13824, 864] |
| pooled_embedding_shape | {'mean': [1, 864], 'max': [1, 864], 'attention': [1, 768]} |

Official Accelerate checkpoint dispatch failed with meta tensors in this environment. We therefore instantiate the official architecture with `get_model(pretrained=False, image_only=True)` and strictly load all image-model tensors from the public safetensors checkpoint, verifying key presence and dimensions. No missing image parameters are permitted.

Full execution log: `logs/colipri_decodability.log`. The runner proceeds automatically from extraction to all 60 primary linear probes and comparison tables.

## Throughput update

The initial serial extractor was interrupted cleanly after 179 cached scans and resumed with four CPU preprocessing workers (`--workers 4`). Workers perform the same QC-selected native DICOM read and official processor; a bounded DataLoader queue overlaps CPU work with single-scan GPU inference. Dense and pooled caching, float32 computation, frozen modules, cohort, labels and seeds are unchanged. Existing atomic scan files are reused. The serial log is preserved in `logs/colipri_decodability_serial.log`.

Resume command uses `CUDA_DEVICE_ORDER=PCI_BUS_ID` so `cuda:3` corresponds to the physical A100 displayed by nvidia-smi. Without this setting, CUDA enumeration placed the initial job on physical GPU 2 (V100). This hardware correction also improves throughput. The initial cached features were generated on V100 and subsequent features on A100 with the same float32 model and checkpoint; minor floating-point hardware differences are possible. Worker-count and GPU scheduling are execution settings and do not change scientific cache provenance.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/run_colipri_decodability.py --device cuda:3 --workers 4
```

Initial measured throughput after queue startup: scan 181 at 38.5 s, 191 at 56.6 s, 201 at 73.3 s (20 new scans / 34.8 s = 1.74 s/scan). The serial run averaged roughly 7–8 s/scan. This initial ~4× improvement suggests ~80 minutes for remaining extraction plus probe training; longer-run throughput may vary with scan size and storage load.
