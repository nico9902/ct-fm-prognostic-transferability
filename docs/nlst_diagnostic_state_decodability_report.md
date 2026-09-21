# NLST diagnostic-state decodability battery

## Executive summary

This experiment asks whether current diagnostic information can be recovered
from frozen Rad-JEPA-3D scan representations using small downstream models. It
is deliberately a current-state experiment: the targets describe findings or
screen interpretation on the same CT, while future cancer diagnosis is not a
target.

The complete battery finished successfully: 4 targets × 5 representations × 5
seeds = 100 runs. Results are close to chance for all four targets. The most
consistent weak signal is for emphysema (`AUROC 0.555 ± 0.002`, sliding-max)
and fibrosis/scarring (`AUROC 0.564 ± 0.016`, gated attention). Isolation
suspicion and ≥4 mm nodule status are not decodable in this setup (`AUROC`
approximately 0.50). These results do not support using the current probes as
strong diagnostic classifiers, but they provide a useful baseline for later
representation and longitudinal-state experiments.

## Scope and data

- Cohort: NLST pilot, 1,000 patients and three nominal screening rounds.
- Input: prepared CT arrays with shape `[1,D,256,256]`, in-plane size 256 × 256,
  and window normalization to `[0,1]`.
- Encoder: frozen Rad-JEPA-3D encoder, 384-dimensional token features, input
  `[B,1,32,256,256]`.
- Usable experiment rows: 2,984 scans from 999 patients. Eleven prepared
  volumes failed preprocessing QC; six additional rows were excluded because
  of missing or ambiguous clinical linkage.
- Patient split: 599 train patients, 200 validation patients, 200 test
  patients; 1,789 / 598 / 597 scans respectively.
- No patient appears in more than one split.
- Depth: minimum 62, mean 169.85, median 158, maximum 545 slices; no usable
  scan had depth below 32. Sliding windows averaged 10.08 per scan, ranging
  from 3 to 34.

The four labels were `isolation_suspicious`, `nodule_ge_4mm`, `emphysema`, and
`fibrosis_or_scarring`. Their usable-cohort prevalences were 39.04%, 38.77%,
41.45%, and 25.07%, respectively.

## Representations and models

The encoder was run once per needed volume/window and kept in a cache. The
cache contains Resize32 scan embeddings and all 32-slice, stride-16 window
embeddings. It was reused for every downstream run.

1. **Resize32:** depth interpolation to 32 slices, Rad-JEPA, mean over tokens
   → `[384]`.
2. **Sliding-Mean:** mean over window embeddings → `[384]`.
3. **Sliding-Max:** element-wise maximum over window embeddings → `[384]`.
4. **Sliding-MeanMax:** concatenate sliding mean and max → `[768]`.
5. **Sliding-GatedAttention:** learned gated-attention MIL aggregation over
   window embeddings → `[384]`, followed by a linear head. This is a learned
   aggregation experiment, not a pure linear probe.

For the first four representations the only trainable component was
`Linear(input_dim,1)`. Gated attention used the specified gated-attention
module plus `Linear(384,1)`. All runs used BCE-with-logits loss, class weights
computed from the training split only, patient-level fixed splits, five seeds,
Lightning early stopping, and deterministic CUDA execution.

Feature standardization was fit on training scans only for each representation.
It was applied to encoder embeddings, not to the already `[0,1]`-normalized CT
inputs.

## Results

Values are mean ± standard deviation across the five seeds. The full requested
metric set—AUROC, AUPRC, MCC, accuracy, F1, precision, recall, specificity,
TN, FP, FN, and TP—is available in `per_seed_results.csv`; aggregate means and
standard deviations are in `summary_results.csv`.

### AUROC / AUPRC / MCC

| Target | Representation | AUROC | AUPRC | MCC |
|---|---|---:|---:|---:|
| isolation_suspicious | Resize32 | 0.505 ± 0.005 | 0.372 ± 0.004 | 0.010 ± 0.013 |
| isolation_suspicious | Sliding-Mean | 0.507 ± 0.007 | 0.371 ± 0.009 | 0.024 ± 0.011 |
| isolation_suspicious | Sliding-Max | 0.499 ± 0.011 | 0.365 ± 0.010 | -0.021 ± 0.029 |
| isolation_suspicious | Sliding-MeanMax | 0.505 ± 0.004 | 0.369 ± 0.003 | -0.001 ± 0.009 |
| isolation_suspicious | Sliding-GatedAttention | 0.508 ± 0.018 | 0.363 ± 0.018 | 0.042 ± 0.035 |
| nodule_ge_4mm | Resize32 | 0.502 ± 0.004 | 0.367 ± 0.004 | 0.007 ± 0.011 |
| nodule_ge_4mm | Sliding-Mean | 0.504 ± 0.006 | 0.363 ± 0.007 | 0.014 ± 0.006 |
| nodule_ge_4mm | Sliding-Max | 0.497 ± 0.012 | 0.358 ± 0.011 | -0.027 ± 0.014 |
| nodule_ge_4mm | Sliding-MeanMax | 0.503 ± 0.004 | 0.361 ± 0.004 | 0.006 ± 0.021 |
| nodule_ge_4mm | Sliding-GatedAttention | 0.506 ± 0.021 | 0.362 ± 0.019 | 0.016 ± 0.041 |
| emphysema | Resize32 | 0.540 ± 0.008 | 0.429 ± 0.005 | 0.061 ± 0.022 |
| emphysema | Sliding-Mean | 0.531 ± 0.006 | 0.431 ± 0.005 | 0.038 ± 0.020 |
| emphysema | Sliding-Max | **0.555 ± 0.002** | 0.437 ± 0.003 | **0.088 ± 0.011** |
| emphysema | Sliding-MeanMax | 0.551 ± 0.008 | 0.435 ± 0.006 | 0.067 ± 0.027 |
| emphysema | Sliding-GatedAttention | 0.544 ± 0.013 | **0.446 ± 0.016** | 0.066 ± 0.023 |
| fibrosis_or_scarring | Resize32 | 0.553 ± 0.007 | 0.352 ± 0.006 | **0.097 ± 0.014** |
| fibrosis_or_scarring | Sliding-Mean | 0.554 ± 0.005 | 0.345 ± 0.005 | 0.090 ± 0.009 |
| fibrosis_or_scarring | Sliding-Max | 0.548 ± 0.004 | **0.358 ± 0.006** | 0.070 ± 0.024 |
| fibrosis_or_scarring | Sliding-MeanMax | 0.542 ± 0.020 | 0.348 ± 0.021 | 0.077 ± 0.032 |
| fibrosis_or_scarring | Sliding-GatedAttention | **0.564 ± 0.016** | 0.351 ± 0.012 | 0.093 ± 0.025 |

Accuracy, F1, precision, recall, specificity, and confusion counts should be
read from the result CSV rather than treated as primary evidence. The
training class weighting and fixed 0.5 threshold make threshold-dependent
metrics secondary; AUROC, AUPRC, and MCC are more informative here.

## Interpretation

### Isolation suspicion

All representations are effectively chance-level. The best AUROC is only
0.508, with MCC near zero. The frozen representation did not recover the
isolation-read suspicion label reliably.

### Nodule ≥4 mm

All representations are also chance-level, with AUROC between 0.497 and 0.506.
This is the most important negative result for the proposed first probe target:
the cached Rad-JEPA features, with these simple scan-level aggregations, do not
show useful recovery of the official ≥4 mm nodule state.

### Emphysema

There is a small but consistent signal above chance. Sliding-Max has the best
AUROC and MCC, while gated attention has the best AUPRC. The effect is weak and
should be considered exploratory, not diagnostic performance.

### Fibrosis or scarring

This target also shows a weak above-chance signal. Gated attention gives the
best AUROC, while Resize32 gives the highest MCC. The AUPRC values should be
compared with the 25% prevalence baseline; they are higher than baseline but
not sufficient to establish robust decodability.

### Representation comparison

There is no universal winner. Sliding-Max is strongest for emphysema, gated
attention is strongest by AUROC for fibrosis, and all methods perform near
chance for suspicion and nodules. The learned aggregator does not provide a
general advantage over simple pooling. The small differences should not be
over-interpreted because all methods use the same fixed split and only five
seeds.

## W&B and reproducibility artifacts

Runs were logged to the W&B project
`nlst-radjepa-diagnostic-decoding`, grouped under
`pilot-1000-diagnostic-state`. Run names encode target, representation, and
seed. The local experiment directory is:

`/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability/`

Important files are:

- `per_seed_results.csv`: 100 rows with all per-seed metrics.
- `summary_results.csv`: mean ± standard deviation by target and representation.
- `patient_splits.csv`: fixed patient-level split assignment.
- `experiment_scan_manifest.csv`: analyzed scans and labels.
- `cache/resize32_embeddings.npy`: cached Resize32 embeddings.
- `cache/sliding_window_embeddings.npy`: cached window embeddings.
- `gated_attention_weights.csv`: learned test-scan attention weights.
- `qc.json`: scan, depth, window, prevalence, and split QC.

## Limitations and next steps

This is a decodability baseline, not a clinical validation study. The cohort is
a selected pilot, the encoder remains frozen, and downstream capacity is
intentionally restricted. Labels are derived from radiologist/read and
abnormality records rather than independent voxel annotations, so read-level
labels may not align exactly with what a scan representation can recover.

The immediate next checks should be: confirm the exact W&B run count and cache
checksums, inspect attention maps for stability, evaluate calibration and
confidence intervals, and test whether a higher-capacity but still
patient-split downstream model changes the negative suspicion/nodule result.
Only after those checks should these representations be promoted to states for
longitudinal next-state prediction.

