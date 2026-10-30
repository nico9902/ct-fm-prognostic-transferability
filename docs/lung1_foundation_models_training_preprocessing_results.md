# LUNG1 CT foundation models: training, preprocessing, and results

**Report date:** 7 October 2026; DINOv3 and statistical tests added 16 October 2026  
**Cohort:** NSCLC-Radiomics (LUNG1), 416 patients (369 deaths, 47 censored)  
**Endpoint:** overall survival from treatment start

## Scope and study design

This report covers the four volumetric CT foundation models evaluated on the common LUNG1 cohort, TANGERINE, Rad-JEPA-3D, COLIPRI, and CT-CLIP, plus DINOv3 ViT-B/16, a 2-D natural-image encoder added as a non-medical reference. The main results are the **completed, selected-checkpoint five-fold comparison** in [`outputs/lung1_bf16_selected`](../outputs/lung1_bf16_selected). It has 25 complete method–encoder combinations and one held-out prediction per patient in each combination. The [statistical tests](#statistical-tests) section adds paired bootstrap tests with Holm correction for all combinations and the two literature baselines. The later sections report the earlier 25-fold frozen-feature and input-crop analyses as separate sensitivity results. These runs use different fitting and aggregation protocols; their scores must not be pooled or interpreted as a single ranking.

The strict imaging audit found 417 globally eligible CTs. The common comparison excludes the one patient without an eligible GTV mask, giving 416 patients for both global and regional analyses. The locked folds stratify patients by event status and early (I/II) versus late (IIIa/IIIb) stage. The primary evaluation measure is Uno's inverse-probability-of-censoring-weighted concordance index through 36 months. See the [cohort audit](lung1_cohort_audit.md) and [locked split file](../configs/lung1_e1_folds.csv).

## Image preprocessing and representation extraction

All main-comparison methods use each patient's same selected **original CT DICOM series**. SimpleITK loads the ordered slices as a three-dimensional HU image, and the extraction checks the slice count against the cohort audit. No outcome label, GTV mask, or common resized tensor enters the encoder. The encoders then receive their own input transforms; dimensions below are depth × height × width, excluding batch and channel axes.

| Encoder and pretraining paradigm | Spatial preparation | HU transform | Input tensor | Frozen global vector |
| --- | --- | --- | ---: | ---: |
| TANGERINE; masked autoencoding | Preserve source orientation and full field of view; B-spline resample to a 256³ grid. Physical output spacing therefore varies by scan. | Clip [−1200, 800], map to [0, 1]. | 256 × 256 × 256 | 2,048; normalized CLS plus mean patch representation. |
| Rad-JEPA-3D; joint-embedding prediction | Reorient to RAS; linearly resize the full array to fixed voxel counts, without fixing physical spacing. | Clip [−1000, 400], map to [0, 1]. | 32 × 256 × 256 | 384; mean of final tokens. |
| COLIPRI; multimodal/multi-objective | Official TorchIO processor: RAS, 2 mm isotropic linear resampling, then central crop or pad; approximately 384 mm per side. | Clamp [−1000, 1000], map to [−1, 1]; pad at minimum intensity. | 192 × 192 × 192 | 768; projected attention pool. |
| CT-CLIP; vision–language contrastive | LPS; trilinear resize to 0.75 × 0.75 × 1.5 mm (x × y × z), then central crop or pad to approximately 360 mm per side. | Clip [−1000, 1000], divide by 1000; pad at −1. | 240 × 480 × 480 | 512; temporal mean, flattened spatial tokens, visual projection and L2 normalization. |
| DINOv3 ViT-B/16; 2-D self-distillation on natural images (LVD-1689M) | LPS; linear resampling along z to 3 mm, full in-plane field of view; each axial slice resized to 224 × 224 (bilinear with antialiasing, as in the official processor). | Clip [−1000, 400], map to [0, 1], replicate to three channels, ImageNet mean/std. | slices × 3 × 224 × 224 | 768; mean over slices of the final-norm CLS token. |

The fixed-voxel and fixed-physical-spacing policies give different anatomical coverage. This is part of each model's implemented input protocol, so a score difference cannot be assigned solely to its pretraining objective. The frozen global vectors were extracted once without survival labels. The frozen-token method instead uses final spatial tokens, stored in float16 and adaptively averaged to at most eight cells per axis: TANGERINE 512 × 1,024, Rad-JEPA-3D 256 × 384, COLIPRI 512 × 768, CT-CLIP 512 × 512, and DINOv3 512 × 768 (tokens × channels; DINOv3 patch tokens form a slices × 14 × 14 grid). The TANGERINE class token and the DINOv3 CLS and four register tokens are excluded; the Rad-JEPA Morton scan is reversed where applicable. The image-level LoRA and partial tuning runs use cached model-specific input tensors from the original CTs. Their cache stores float16 tensors; neural training uses BF16 mixed precision.

The CT-CLIP checkpoint came from a public mirror. Its identity with the authors' gated checkpoint has not been verified; the [checkpoint metadata](../external/ctclip/checkpoint_metadata.json) records provenance. The [extraction implementation](../scripts/extract_lung1_e1_embeddings.py), [token implementation](../scripts/extract_lung1_e2_tokens.py), and [input-cache implementation](../scripts/cache_lung1_e2_inputs.py) define the exact transforms.

## Training and validation setup for the main comparison

The main comparison uses repetition zero of the locked five-fold outer split. Within each outer training set, locked inner fold zero is a validation set: approximately 221 training, 111 validation, and 83–84 untouched test patients per fold. The same patient assignments are used for every model and method. Training-only data fit all feature scaling, Cox baselines, and censoring distributions. Each candidate is evaluated once on its outer test fold after validation selection. The five test folds yield 416 unique held-out predictions per combination.

| Method | Fitted components and fixed training budget | Validation choice |
| --- | --- | --- |
| Frozen linear Cox | Frozen global vector; training-set standardization; 32-component PCA; L2-regularized Cox fit with candidate penalties 0.01, 0.1, 1.0. | Penalty with highest validation Uno C; retains that training-only fit. |
| Frozen MLP Cox | Frozen global vector; training-set standardization; one 32-unit hidden layer, dropout 0.1, AdamW learning rate 3×10⁻⁴, weight decay 10⁻⁴; 100 epochs. | Epoch with highest validation Uno C. |
| Frozen-token survival query adapter (SQA) | Training-set token-channel normalization; 128-dimensional projection, fixed 3D Fourier positions, four learned queries, four-head cross-attention and Cox head; dropout 0.1, AdamW learning rate 3×10⁻⁴, weight decay 10⁻⁴; 40 epochs. | Epoch with highest validation Uno C. |
| LoRA | Re-encode the original CT; rank 8, alpha 16, learning rate 10⁻⁴, weight decay 10⁻⁴; 10 epochs. A new normalization and Cox head is also trained. | Epoch with highest validation Uno C. |
| Partial encoder tuning | Re-encode the original CT; unfreeze the last selected encoder block, learning rate 10⁻⁵, weight decay 10⁻⁴; 10 epochs. A new normalization and Cox head is also trained. | Epoch with highest validation Uno C. |

Neural methods optimize Efron-tie Cox partial likelihood in PyTorch Lightning on an A100 with BF16 mixed precision; risk-set reductions and loss stay in FP32. Linear Cox and PCA use FP32/FP64 computation. All neural runs complete their fixed epoch budget, save the earliest epoch tied for maximum validation Uno C, and restore it for the single test pass. Per-fold Breslow baseline hazards are estimated from training patients to produce survival curves. TANGERINE, Rad-JEPA-3D, COLIPRI, and DINOv3 adapt their last ViT, Mamba, EVA, and ViT block, respectively. DINOv3 encodes all slices of the volume and averages the slice CLS tokens before the head; it needs transformers ≥ 4.56 and runs from `.venv-dinov3`. For CT-CLIP, the vector quantizer blocks gradients from the exported visual vector into those blocks, so LoRA and partial tuning adapt the final visual projection instead. Its trainable capacity is therefore different from the other encoders. Full encoder fine-tuning was not part of this completed comparison. The [selected-checkpoint protocol](lung1_bf16_selected_protocol.md) and the per-run `protocol.json` files give reproducible settings and hashes.

## Main results: selected-checkpoint five-fold comparison

The main tables report **pooled held-out metrics across 416 unique patients**, not the mean of the five fold metrics. For pooled discrimination, risk is negative log predicted survival at 36 months; horizon-specific cumulative hazards give the 1-, 2-, and 3-year AUCs. This avoids treating raw Cox scores from independently fitted folds as if they shared an origin. Higher Uno C and AUC are better. Lower Brier score and integrated Brier score (IBS) are better. The two-year calibration offset is mean predicted minus observed survival, so zero is ideal; a calibration slope near one is desirable. The Uno C intervals are 95% percentile intervals from 500 paired patient bootstrap resamples (see [Statistical tests](#statistical-tests)); the other columns are point estimates.

### Discrimination

The column p<sub>Holm</sub> (C > 0.5) tests pooled Uno C against chance, Holm-adjusted over the 25 FM conditions and the two baselines.

| Method | Encoder | Uno C, 36 m (95% CI) | p<sub>Holm</sub> (C > 0.5) | AUC, 1 y | AUC, 2 y | AUC, 3 y |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear Cox | COLIPRI | 0.619 (0.582–0.653) | <0.001 | 0.667 | 0.647 | 0.670 |
| Linear Cox | TANGERINE | 0.595 (0.562–0.630) | <0.001 | 0.620 | 0.607 | 0.643 |
| Linear Cox | Rad-JEPA-3D | 0.528 (0.491–0.564) | 1.000 | 0.529 | 0.540 | 0.544 |
| Linear Cox | CT-CLIP | 0.513 (0.477–0.553) | 1.000 | 0.509 | 0.528 | 0.533 |
| Linear Cox | DINOv3 | 0.546 (0.509–0.576) | 0.141 | 0.562 | 0.557 | 0.572 |
| MLP Cox | COLIPRI | 0.616 (0.584–0.647) | <0.001 | 0.663 | 0.637 | 0.667 |
| MLP Cox | TANGERINE | 0.611 (0.578–0.644) | <0.001 | 0.662 | 0.636 | 0.653 |
| MLP Cox | Rad-JEPA-3D | 0.530 (0.495–0.562) | 1.000 | 0.527 | 0.555 | 0.548 |
| MLP Cox | CT-CLIP | 0.485 (0.449–0.520) | 1.000 | 0.488 | 0.480 | 0.479 |
| MLP Cox | DINOv3 | 0.561 (0.529–0.593) | 0.006 | 0.563 | 0.580 | 0.605 |
| SQA | COLIPRI | 0.605 (0.566–0.639) | <0.001 | 0.648 | 0.629 | 0.660 |
| SQA | TANGERINE | 0.608 (0.577–0.641) | <0.001 | 0.648 | 0.623 | 0.650 |
| SQA | Rad-JEPA-3D | 0.499 (0.463–0.534) | 1.000 | 0.530 | 0.506 | 0.489 |
| SQA | CT-CLIP | 0.501 (0.466–0.535) | 1.000 | 0.509 | 0.511 | 0.504 |
| SQA | DINOv3 | 0.533 (0.499–0.566) | 1.000 | 0.546 | 0.533 | 0.548 |
| LoRA | COLIPRI | 0.535 (0.501–0.572) | 0.922 | 0.521 | 0.529 | 0.552 |
| LoRA | TANGERINE | 0.520 (0.486–0.555) | 1.000 | 0.497 | 0.523 | 0.548 |
| LoRA | Rad-JEPA-3D | 0.492 (0.455–0.527) | 1.000 | 0.486 | 0.466 | 0.478 |
| LoRA | CT-CLIP | 0.517 (0.482–0.556) | 1.000 | 0.510 | 0.531 | 0.551 |
| LoRA | DINOv3 | 0.489 (0.452–0.524) | 1.000 | 0.482 | 0.462 | 0.470 |
| Partial tuning | COLIPRI | 0.490 (0.460–0.528) | 1.000 | 0.525 | 0.476 | 0.464 |
| Partial tuning | TANGERINE | 0.509 (0.474–0.544) | 1.000 | 0.494 | 0.501 | 0.515 |
| Partial tuning | Rad-JEPA-3D | 0.499 (0.465–0.534) | 1.000 | 0.481 | 0.513 | 0.512 |
| Partial tuning | CT-CLIP | 0.514 (0.481–0.551) | 1.000 | 0.465 | 0.533 | 0.551 |
| Partial tuning | DINOv3 | 0.506 (0.471–0.540) | 1.000 | 0.522 | 0.481 | 0.508 |
| Baseline Cox | Stage + tumor volume | 0.557 (0.527–0.591) | 0.012 | — | — | — |
| Baseline Cox | GTV radiomics | 0.570 (0.538–0.602) | <0.001 | — | — | — |

The baselines are described in the [baseline report](lung1_bf16_selected_baselines_report.md); their AUCs are 0.560 / 0.592 / 0.594 (stage + volume) and 0.614 / 0.596 / 0.601 (radiomics).

### Frozen linear Cox: full-follow-up C-indices (five-fold means)

These scores use the same held-out predictions but are **means of the five test-fold concordance indices**, not pooled scores. Harrell's C-index compares observed survival times over full follow-up without censoring weights. Uno's C-index includes deaths through the latest observed death at 3,653 days and uses each fold's outer-training censoring distribution. Neither method was refitted. The five-fold means differ from the pooled 36-month Uno C values in the main discrimination table above. All five rows were recomputed by `scripts/test_lung1_bf16_selected_stats.py`; the four 3-D encoders match the earlier values within 0.001.

| Foundation model | Harrell's C-index | Full-follow-up Uno C-index |
| --- | ---: | ---: |
| COLIPRI | **0.618** | 0.616 |
| TANGERINE | 0.600 | 0.600 |
| Rad-JEPA-3D | 0.533 | 0.532 |
| CT-CLIP | 0.513 | 0.512 |
| DINOv3 | 0.552 | 0.552 |

### Survival prediction error and calibration

| Method | Encoder | Brier, 1 y | Brier, 2 y | Brier, 3 y | IBS, 3–36 m | 2 y offset | Calibration slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear Cox | COLIPRI | 0.214 | 0.231 | 0.201 | 0.204 | 0.011 | 0.225 |
| Linear Cox | TANGERINE | 0.222 | 0.243 | 0.208 | 0.212 | 0.010 | 0.197 |
| Linear Cox | Rad-JEPA-3D | 0.231 | 0.243 | 0.217 | 0.217 | 0.014 | 0.260 |
| Linear Cox | CT-CLIP | 0.234 | 0.246 | 0.220 | 0.219 | 0.020 | 0.211 |
| Linear Cox | DINOv3 | 0.233 | 0.249 | 0.220 | 0.221 | 0.011 | 0.112 |
| MLP Cox | COLIPRI | 0.213 | 0.234 | 0.204 | 0.205 | -0.013 | 0.247 |
| MLP Cox | TANGERINE | 0.225 | 0.249 | 0.220 | 0.216 | -0.022 | 0.057 |
| MLP Cox | Rad-JEPA-3D | 0.235 | 0.243 | 0.217 | 0.218 | 0.005 | 0.279 |
| MLP Cox | CT-CLIP | 0.247 | 0.268 | 0.240 | 0.235 | -0.008 | 0.030 |
| MLP Cox | DINOv3 | 0.256 | 0.274 | 0.242 | 0.240 | -0.056 | 0.047 |
| SQA | COLIPRI | 0.218 | 0.238 | 0.204 | 0.209 | -0.007 | 0.187 |
| SQA | TANGERINE | 0.238 | 0.264 | 0.226 | 0.226 | 0.002 | 0.054 |
| SQA | Rad-JEPA-3D | 0.261 | 0.282 | 0.255 | 0.249 | 0.023 | 0.027 |
| SQA | CT-CLIP | 0.247 | 0.265 | 0.239 | 0.235 | -0.018 | 0.052 |
| SQA | DINOv3 | 0.278 | 0.313 | 0.269 | 0.269 | -0.025 | 0.010 |
| LoRA | COLIPRI | 0.235 | 0.247 | 0.219 | 0.219 | 0.021 | 0.319 |
| LoRA | TANGERINE | 0.230 | 0.241 | 0.214 | 0.215 | 0.017 | 0.716 |
| LoRA | Rad-JEPA-3D | 0.232 | 0.248 | 0.220 | 0.220 | 0.018 | 0.103 |
| LoRA | CT-CLIP | 0.246 | 0.263 | 0.228 | 0.232 | 0.002 | 0.064 |
| LoRA | DINOv3 | 0.232 | 0.248 | 0.220 | 0.220 | 0.016 | -0.081 |
| Partial tuning | COLIPRI | 0.244 | 0.273 | 0.249 | 0.240 | 0.045 | -0.059 |
| Partial tuning | TANGERINE | 0.230 | 0.242 | 0.215 | 0.215 | 0.016 | 0.476 |
| Partial tuning | Rad-JEPA-3D | 0.235 | 0.246 | 0.220 | 0.220 | 0.024 | 0.004 |
| Partial tuning | CT-CLIP | 0.270 | 0.273 | 0.238 | 0.245 | -0.007 | 0.023 |
| Partial tuning | DINOv3 | 0.230 | 0.247 | 0.218 | 0.218 | 0.018 | -0.072 |

The highest pooled 36-month Uno C is **COLIPRI frozen linear Cox, 0.619**; its IBS is also the lowest, **0.204**. Among all methods, the best TANGERINE Uno C is the frozen MLP (0.611); the best Rad-JEPA-3D is frozen MLP (0.530); the best CT-CLIP is LoRA (0.517); the best DINOv3 is frozen MLP (0.561). None of the LoRA or partial-tuning runs exceeds its encoder's frozen linear Uno C by a large margin, and COLIPRI partial tuning falls to 0.490. SQA raises TANGERINE's pooled Uno C from 0.595 to 0.608, while it lowers the other four encoders' scores relative to their frozen linear probes. Some high-rank results still have poor calibration slopes: for example, COLIPRI frozen linear has a pooled slope of 0.225, and all DINOv3 rungs have pooled slopes at or below 0.112. The mean calibration offset alone should not be read as good individual prediction calibration.

The [main output directory](../outputs/lung1_bf16_selected) contains `summary.json`, `fold_metrics.csv`, `out_of_fold_predictions.csv`, and `protocol.json` under each `<method>/<encoder>` directory. `summary.json` also reports the mean of five fold metrics. The individual fold files include selected penalty or epoch, training and validation counts, training time, and, for neural fits, trainable parameters and peak GPU memory. The `COMPLETE` and `COMPLETE_dinov3` markers and all 25 summaries were present when this report was written. DINOv3-specific details are in the [DINOv3 report](lung1_dinov3_bf16_selected_report.md).

## Statistical tests

All tests use pooled Uno C at 36 months over the 416 out-of-fold predictions, with the same 500 paired patient bootstrap resamples (seed 20261015) as the [baseline report](lung1_bf16_selected_baselines_report.md) and the [DINOv3 report](lung1_dinov3_bf16_selected_report.md). The censoring curve is fixed to the full cohort. Because each resample keeps the same patients across conditions, differences account for the correlation between conditions. Intervals are bootstrap percentile intervals. p-values are two-sided Wald tests, z = difference / bootstrap SE; 500 resamples are too few for direct bootstrap p-values under multiplicity correction. Holm's correction is applied separately within each family below. All tests are conditional on the fitted fold models and the locked split: they quantify uncertainty from the patient sample, not from refitting or from the choice of fold. They are exploratory; the study had no pre-registered primary contrast.

### Do encoders differ within each rung?

The omnibus test is a 4-df Wald χ² on the four encoder contrasts against TANGERINE, using their bootstrap covariance. H0: all five encoders have equal pooled Uno C. The Friedman test is a non-pooled complement: it ranks the five encoders within each test fold (n = 5 folds), so it does not depend on pooling separately fitted folds, but it has low power. Holm is applied over the five rungs for each test.

| Rung | Wald χ² (4 df) | p | p<sub>Holm</sub> | Friedman χ² (4 df) | p<sub>Holm</sub> | Mean fold rank (1 = best): COLIPRI / TANGERINE / DINOv3 / Rad-JEPA-3D / CT-CLIP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Linear Cox | 27.0 | <0.001 | <0.001 | 13.1 | 0.043 | 1.6 / 1.6 / 3.8 / 4.0 / 4.0 |
| MLP Cox | 46.8 | <0.001 | <0.001 | 15.8 | 0.016 | 1.4 / 2.0 / 3.0 / 3.6 / 5.0 |
| SQA | 47.5 | <0.001 | <0.001 | 7.0 | 0.401 | 2.0 / 2.0 / 3.6 / 3.4 / 4.0 |
| LoRA | 5.8 | 0.218 | 0.436 | 4.3 | 0.729 | 2.2 / 2.2 / 3.6 / 3.6 / 3.4 |
| Partial tuning | 1.1 | 0.901 | 0.901 | 0.3 | 0.988 | 3.0 / 2.8 / 2.8 / 3.2 / 3.2 |

The encoders differ in the three frozen rungs (pooled Wald p<sub>Holm</sub> < 0.001); the fold-level Friedman test agrees for Linear and MLP but not for SQA. No encoder difference is detectable after LoRA or partial tuning, where all encoders are near chance.

### Pairwise encoder contrasts

Row minus column, pooled Uno C difference [95% CI]; Holm over all 50 pairwise contrasts (10 pairs × 5 rungs). Bold: p<sub>Holm</sub> < 0.05. Shown for the three frozen rungs; every LoRA and partial-tuning pair has p<sub>Holm</sub> = 1.000, and all 50 are in the JSON file.

In summary, COLIPRI and TANGERINE are never distinguishable from each other, and each is significantly above Rad-JEPA-3D and CT-CLIP in almost every frozen rung. DINOv3 is in between. It is significantly below COLIPRI only in the Linear rung and below TANGERINE only in SQA. It is significantly above CT-CLIP only in the MLP rung, and never distinguishable from Rad-JEPA-3D. Several DINOv3 contrasts whose unadjusted intervals exclude zero, for example TANGERINE − DINOv3 in Linear and MLP, are not significant after Holm correction.

**Linear Cox**

| Contrast | Difference [95% CI]; p<sub>Holm</sub> |
| --- | --- |
| COLIPRI − TANGERINE | +0.024 [-0.016, +0.064]; p<sub>Holm</sub> 1.000 |
| COLIPRI − DINOv3 | **+0.073 [+0.032, +0.116]; p<sub>Holm</sub> 0.038** |
| COLIPRI − Rad-JEPA-3D | **+0.091 [+0.042, +0.132]; p<sub>Holm</sub> 0.003** |
| COLIPRI − CT-CLIP | **+0.106 [+0.057, +0.152]; p<sub>Holm</sub> <0.001** |
| TANGERINE − DINOv3 | +0.050 [+0.013, +0.085]; p<sub>Holm</sub> 0.274 |
| TANGERINE − Rad-JEPA-3D | +0.067 [+0.020, +0.105]; p<sub>Holm</sub> 0.069 |
| TANGERINE − CT-CLIP | **+0.082 [+0.037, +0.124]; p<sub>Holm</sub> 0.007** |
| DINOv3 − Rad-JEPA-3D | +0.017 [-0.030, +0.059]; p<sub>Holm</sub> 1.000 |
| DINOv3 − CT-CLIP | +0.033 [-0.013, +0.073]; p<sub>Holm</sub> 1.000 |
| Rad-JEPA-3D − CT-CLIP | +0.016 [-0.027, +0.057]; p<sub>Holm</sub> 1.000 |

**MLP Cox**

| Contrast | Difference [95% CI]; p<sub>Holm</sub> |
| --- | --- |
| COLIPRI − TANGERINE | +0.005 [-0.030, +0.045]; p<sub>Holm</sub> 1.000 |
| COLIPRI − DINOv3 | +0.055 [+0.014, +0.098]; p<sub>Holm</sub> 0.330 |
| COLIPRI − Rad-JEPA-3D | **+0.086 [+0.042, +0.132]; p<sub>Holm</sub> 0.006** |
| COLIPRI − CT-CLIP | **+0.131 [+0.086, +0.179]; p<sub>Holm</sub> <0.001** |
| TANGERINE − DINOv3 | +0.050 [+0.016, +0.087]; p<sub>Holm</sub> 0.165 |
| TANGERINE − Rad-JEPA-3D | **+0.081 [+0.041, +0.119]; p<sub>Holm</sub> 0.002** |
| TANGERINE − CT-CLIP | **+0.126 [+0.084, +0.167]; p<sub>Holm</sub> <0.001** |
| DINOv3 − Rad-JEPA-3D | +0.030 [-0.010, +0.074]; p<sub>Holm</sub> 1.000 |
| DINOv3 − CT-CLIP | **+0.076 [+0.030, +0.118]; p<sub>Holm</sub> 0.032** |
| Rad-JEPA-3D − CT-CLIP | +0.045 [+0.004, +0.087]; p<sub>Holm</sub> 0.938 |

**SQA**

| Contrast | Difference [95% CI]; p<sub>Holm</sub> |
| --- | --- |
| COLIPRI − TANGERINE | -0.003 [-0.046, +0.038]; p<sub>Holm</sub> 1.000 |
| COLIPRI − DINOv3 | +0.072 [+0.026, +0.114]; p<sub>Holm</sub> 0.083 |
| COLIPRI − Rad-JEPA-3D | **+0.105 [+0.060, +0.151]; p<sub>Holm</sub> <0.001** |
| COLIPRI − CT-CLIP | **+0.104 [+0.056, +0.148]; p<sub>Holm</sub> <0.001** |
| TANGERINE − DINOv3 | **+0.075 [+0.038, +0.115]; p<sub>Holm</sub> 0.004** |
| TANGERINE − Rad-JEPA-3D | **+0.109 [+0.068, +0.148]; p<sub>Holm</sub> <0.001** |
| TANGERINE − CT-CLIP | **+0.107 [+0.065, +0.146]; p<sub>Holm</sub> <0.001** |
| DINOv3 − Rad-JEPA-3D | +0.034 [-0.012, +0.080]; p<sub>Holm</sub> 1.000 |
| DINOv3 − CT-CLIP | +0.032 [-0.020, +0.081]; p<sub>Holm</sub> 1.000 |
| Rad-JEPA-3D − CT-CLIP | -0.002 [-0.046, +0.043]; p<sub>Holm</sub> 1.000 |

### Foundation models versus literature baselines

Condition minus baseline [95% CI]; Holm over the 50 contrasts (25 conditions × 2 baselines). Bold: p<sub>Holm</sub> < 0.05.

| Condition | − Stage + volume | − Radiomics |
| --- | --- | --- |
| Linear Cox COLIPRI | +0.062 [+0.014, +0.104]; p<sub>Holm</sub> 0.243 | +0.049 [+0.005, +0.087]; p<sub>Holm</sub> 0.487 |
| Linear Cox TANGERINE | +0.038 [-0.009, +0.080]; p<sub>Holm</sub> 1.000 | +0.026 [-0.018, +0.070]; p<sub>Holm</sub> 1.000 |
| Linear Cox Rad-JEPA-3D | -0.028 [-0.071, +0.016]; p<sub>Holm</sub> 1.000 | -0.041 [-0.085, +0.001]; p<sub>Holm</sub> 1.000 |
| Linear Cox CT-CLIP | -0.044 [-0.091, -0.001]; p<sub>Holm</sub> 1.000 | -0.057 [-0.104, -0.006]; p<sub>Holm</sub> 0.610 |
| Linear Cox DINOv3 | -0.011 [-0.063, +0.034]; p<sub>Holm</sub> 1.000 | -0.024 [-0.072, +0.020]; p<sub>Holm</sub> 1.000 |
| MLP Cox COLIPRI | +0.059 [+0.013, +0.098]; p<sub>Holm</sub> 0.245 | +0.046 [+0.006, +0.082]; p<sub>Holm</sub> 0.511 |
| MLP Cox TANGERINE | +0.054 [+0.011, +0.093]; p<sub>Holm</sub> 0.360 | +0.041 [-0.002, +0.082]; p<sub>Holm</sub> 1.000 |
| MLP Cox Rad-JEPA-3D | -0.027 [-0.074, +0.010]; p<sub>Holm</sub> 1.000 | -0.039 [-0.083, +0.003]; p<sub>Holm</sub> 1.000 |
| MLP Cox CT-CLIP | -0.072 [-0.122, -0.022]; p<sub>Holm</sub> 0.122 | **-0.085 [-0.133, -0.036]; p<sub>Holm</sub> 0.044** |
| MLP Cox DINOv3 | +0.004 [-0.040, +0.047]; p<sub>Holm</sub> 1.000 | -0.009 [-0.051, +0.037]; p<sub>Holm</sub> 1.000 |
| SQA COLIPRI | +0.048 [-0.001, +0.089]; p<sub>Holm</sub> 0.809 | +0.035 [-0.005, +0.076]; p<sub>Holm</sub> 1.000 |
| SQA TANGERINE | +0.051 [+0.009, +0.088]; p<sub>Holm</sub> 0.384 | +0.038 [-0.000, +0.075]; p<sub>Holm</sub> 1.000 |
| SQA Rad-JEPA-3D | -0.058 [-0.105, -0.018]; p<sub>Holm</sub> 0.382 | -0.070 [-0.118, -0.026]; p<sub>Holm</sub> 0.128 |
| SQA CT-CLIP | -0.056 [-0.101, -0.013]; p<sub>Holm</sub> 0.511 | -0.069 [-0.115, -0.021]; p<sub>Holm</sub> 0.235 |
| SQA DINOv3 | -0.024 [-0.067, +0.017]; p<sub>Holm</sub> 1.000 | -0.037 [-0.082, +0.008]; p<sub>Holm</sub> 1.000 |
| LoRA COLIPRI | -0.022 [-0.070, +0.022]; p<sub>Holm</sub> 1.000 | -0.034 [-0.079, +0.013]; p<sub>Holm</sub> 1.000 |
| LoRA TANGERINE | -0.037 [-0.081, +0.002]; p<sub>Holm</sub> 1.000 | -0.050 [-0.092, -0.003]; p<sub>Holm</sub> 0.600 |
| LoRA Rad-JEPA-3D | -0.065 [-0.112, -0.017]; p<sub>Holm</sub> 0.318 | **-0.078 [-0.124, -0.034]; p<sub>Holm</sub> 0.047** |
| LoRA CT-CLIP | -0.039 [-0.088, +0.006]; p<sub>Holm</sub> 1.000 | -0.052 [-0.095, -0.004]; p<sub>Holm</sub> 0.724 |
| LoRA DINOv3 | -0.068 [-0.113, -0.020]; p<sub>Holm</sub> 0.140 | **-0.081 [-0.128, -0.033]; p<sub>Holm</sub> 0.041** |
| Partial tuning COLIPRI | -0.066 [-0.117, -0.014]; p<sub>Holm</sub> 0.382 | -0.079 [-0.124, -0.026]; p<sub>Holm</sub> 0.106 |
| Partial tuning TANGERINE | -0.048 [-0.088, -0.007]; p<sub>Holm</sub> 0.610 | -0.060 [-0.104, -0.014]; p<sub>Holm</sub> 0.257 |
| Partial tuning Rad-JEPA-3D | -0.058 [-0.101, -0.013]; p<sub>Holm</sub> 0.419 | -0.070 [-0.117, -0.025]; p<sub>Holm</sub> 0.136 |
| Partial tuning CT-CLIP | -0.043 [-0.089, +0.003]; p<sub>Holm</sub> 1.000 | -0.055 [-0.102, -0.009]; p<sub>Holm</sub> 0.610 |
| Partial tuning DINOv3 | -0.051 [-0.098, -0.011]; p<sub>Holm</sub> 0.724 | -0.064 [-0.108, -0.020]; p<sub>Holm</sub> 0.245 |

No frozen FM condition is significantly better than either baseline after correction. The unadjusted intervals for COLIPRI Linear and MLP exclude zero against both baselines (as reported in the [baseline report](lung1_bf16_selected_baselines_report.md)), but the smallest Holm-adjusted p-value among them is 0.243. Three conditions are significantly worse than radiomics after correction: CT-CLIP MLP, Rad-JEPA-3D LoRA and DINOv3 LoRA.

**Per-rung Holm families (added 17 October 2026).** To match the E3 family structure, Holm was also applied within each rung, to its 10 contrasts (5 FMs × 2 baselines), using the same p-values. These families were chosen after the results were known. Only one positive contrast reaches p<sub>Holm</sub> < 0.05: **MLP COLIPRI − stage + volume, +0.059, p<sub>Holm</sub> 0.049**. Linear COLIPRI − stage + volume reaches 0.059. More negative contrasts pass, e.g. MLP CT-CLIP, SQA Rad-JEPA-3D, LoRA Rad-JEPA-3D and DINOv3, and partial tuning COLIPRI, TANGERINE and Rad-JEPA-3D against radiomics. The full table, with both Holm columns, is `outputs/lung1_bf16_selected_baselines/statistical_tests_fm_vs_baselines.csv`. The conclusion is unchanged. The best frozen FMs are at most marginally above stage + volume and not above radiomics, and fine-tuned rungs are below the baselines.

### Adapted rungs versus the frozen linear probe

Rung minus Linear Cox of the same encoder [95% CI]; Holm over the 20 contrasts. Bold: p<sub>Holm</sub> < 0.05.

| Encoder | MLP − Linear | SQA − Linear | LoRA − Linear | Partial − Linear |
| --- | --- | --- | --- | --- |
| COLIPRI | -0.003 [-0.030, +0.023]; p<sub>Holm</sub> 1.000 | -0.014 [-0.041, +0.011]; p<sub>Holm</sub> 1.000 | **-0.084 [-0.122, -0.039]; p<sub>Holm</sub> 0.001** | **-0.129 [-0.178, -0.080]; p<sub>Holm</sub> <0.001** |
| TANGERINE | +0.015 [-0.008, +0.041]; p<sub>Holm</sub> 1.000 | +0.012 [-0.008, +0.038]; p<sub>Holm</sub> 1.000 | **-0.076 [-0.115, -0.034]; p<sub>Holm</sub> 0.003** | **-0.086 [-0.125, -0.040]; p<sub>Holm</sub> <0.001** |
| Rad-JEPA-3D | +0.002 [-0.025, +0.026]; p<sub>Holm</sub> 1.000 | -0.029 [-0.066, +0.006]; p<sub>Holm</sub> 1.000 | -0.037 [-0.087, +0.011]; p<sub>Holm</sub> 1.000 | -0.029 [-0.074, +0.016]; p<sub>Holm</sub> 1.000 |
| CT-CLIP | -0.028 [-0.057, +0.001]; p<sub>Holm</sub> 1.000 | -0.012 [-0.057, +0.033]; p<sub>Holm</sub> 1.000 | +0.004 [-0.025, +0.036]; p<sub>Holm</sub> 1.000 | +0.001 [-0.033, +0.040]; p<sub>Holm</sub> 1.000 |
| DINOv3 | +0.015 [-0.016, +0.048]; p<sub>Holm</sub> 1.000 | -0.013 [-0.049, +0.025]; p<sub>Holm</sub> 1.000 | -0.057 [-0.103, -0.015]; p<sub>Holm</sub> 0.207 | -0.040 [-0.083, +0.003]; p<sub>Holm</sub> 0.895 |

Neither MLP nor SQA improves significantly over the linear probe for any encoder. LoRA and partial tuning are significantly worse than the linear probe for TANGERINE and COLIPRI, the two encoders with a usable frozen signal. For Rad-JEPA-3D, CT-CLIP and DINOv3 the changes are not significant after correction, since their linear probes are already weak (0.513–0.546). The DINOv3 LoRA decrease, −0.057, has an unadjusted interval that excludes zero, but p<sub>Holm</sub> is 0.207.

### Reproduction

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/test_lung1_bf16_selected_stats.py
```

The script writes `outputs/lung1_bf16_selected/statistical_tests.json` with every point estimate, bootstrap SE, interval, raw and Holm-adjusted p-value, the omnibus and Friedman tests, and the fold-level Harrell and full-follow-up Uno C. It takes about three minutes on CPU.

## Earlier frozen-probe and crop sensitivity results

Before the selected-checkpoint comparison, the frozen-feature experiment used five repetitions of five outer folds, three inner folds to tune the Cox penalty, and **mean fold metrics** as point estimates. It tested original CT and a separate GTV-preserving thorax crop, each with PCA32 Cox and full-embedding ridge Cox. The crop is a 1 mm isotropic float32 HU bounding box around both lungs and the GTV with a 10 mm margin; all 416 stored crops retain the entire GTV. Each encoder's clipping, normalization, resampling, and final fixed-size crop/pad is then applied once. GTV annotation determines the crop field of view, and GTV retention in every final encoder tensor was not audited. The full ridge probe uses Efron Cox with L-BFGS and a penalty grid of 0.01, 0.1, 1, 10; the earlier PCA32 probe uses 0.01, 0.1, 1. Thus the probe comparison changes more than feature dimension.

The following tables expose the earlier run's evaluated survival metrics. They are **mean outer-fold scores**, unlike the pooled main tables. Uno C intervals come from 500 paired patient-cluster bootstrap resamples in the [condition comparison](../outputs/lung1_e1_condition_comparison/paired_uno_c.json); models were not refitted in resamples. The other columns are point estimates. These results are sensitivity analyses, and the repeated-fold intervals are conditional on the fitted models and locked folds.

### Earlier discrimination

| CT source / probe | Encoder | Uno C, 36 m (95% interval) | AUC, 1 y | AUC, 2 y | AUC, 3 y |
| --- | --- | ---: | ---: | ---: | ---: |
| Original / PCA32 | COLIPRI | 0.634 (0.602–0.670) | 0.677 | 0.675 | 0.719 |
| Original / PCA32 | TANGERINE | 0.613 (0.584–0.644) | 0.638 | 0.641 | 0.673 |
| Original / PCA32 | Rad-JEPA-3D | 0.550 (0.518–0.582) | 0.561 | 0.567 | 0.573 |
| Original / PCA32 | CT-CLIP | 0.529 (0.497–0.561) | 0.537 | 0.557 | 0.547 |
| Original / full ridge | COLIPRI | 0.633 (0.602–0.667) | 0.679 | 0.669 | 0.714 |
| Original / full ridge | TANGERINE | 0.634 (0.600–0.663) | 0.674 | 0.663 | 0.687 |
| Original / full ridge | Rad-JEPA-3D | 0.530 (0.499–0.558) | 0.526 | 0.548 | 0.546 |
| Original / full ridge | CT-CLIP | 0.533 (0.502–0.562) | 0.541 | 0.552 | 0.554 |
| Crop / PCA32 | COLIPRI | 0.631 (0.600–0.666) | 0.665 | 0.674 | 0.712 |
| Crop / PCA32 | TANGERINE | 0.616 (0.585–0.647) | 0.654 | 0.651 | 0.674 |
| Crop / PCA32 | Rad-JEPA-3D | 0.564 (0.530–0.599) | 0.603 | 0.584 | 0.581 |
| Crop / PCA32 | CT-CLIP | 0.578 (0.549–0.608) | 0.614 | 0.606 | 0.611 |
| Crop / full ridge | COLIPRI | 0.635 (0.603–0.670) | 0.673 | 0.675 | 0.718 |
| Crop / full ridge | TANGERINE | 0.626 (0.589–0.655) | 0.665 | 0.657 | 0.688 |
| Crop / full ridge | Rad-JEPA-3D | 0.576 (0.544–0.609) | 0.619 | 0.602 | 0.599 |
| Crop / full ridge | CT-CLIP | 0.589 (0.558–0.622) | 0.615 | 0.623 | 0.613 |

### Earlier prediction error and calibration

| CT source / probe | Encoder | Brier, 1 y | Brier, 2 y | Brier, 3 y | IBS, 3–36 m | 2 y offset | Calibration slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original / PCA32 | COLIPRI | 0.214 | 0.226 | 0.194 | 0.201 | -0.010 | 1.328 |
| Original / PCA32 | TANGERINE | 0.215 | 0.230 | 0.201 | 0.203 | -0.009 | 1.169 |
| Original / PCA32 | Rad-JEPA-3D | 0.229 | 0.242 | 0.216 | 0.215 | -0.006 | 0.640 |
| Original / PCA32 | CT-CLIP | 0.230 | 0.242 | 0.219 | 0.216 | -0.004 | 0.374 |
| Original / full ridge | COLIPRI | 0.210 | 0.227 | 0.192 | 0.200 | -0.009 | 0.713 |
| Original / full ridge | TANGERINE | 0.224 | 0.243 | 0.211 | 0.212 | -0.047 | 0.527 |
| Original / full ridge | Rad-JEPA-3D | 0.245 | 0.257 | 0.233 | 0.231 | -0.008 | 0.457 |
| Original / full ridge | CT-CLIP | 0.257 | 0.271 | 0.242 | 0.242 | -0.021 | 0.270 |
| Crop / PCA32 | COLIPRI | 0.214 | 0.225 | 0.194 | 0.201 | -0.008 | 1.310 |
| Crop / PCA32 | TANGERINE | 0.212 | 0.228 | 0.202 | 0.202 | -0.010 | 0.896 |
| Crop / PCA32 | Rad-JEPA-3D | 0.222 | 0.243 | 0.221 | 0.214 | -0.002 | 0.484 |
| Crop / PCA32 | CT-CLIP | 0.223 | 0.238 | 0.214 | 0.212 | -0.009 | 0.563 |
| Crop / full ridge | COLIPRI | 0.212 | 0.225 | 0.191 | 0.200 | -0.011 | 0.743 |
| Crop / full ridge | TANGERINE | 0.213 | 0.230 | 0.200 | 0.203 | -0.028 | 0.736 |
| Crop / full ridge | Rad-JEPA-3D | 0.223 | 0.244 | 0.220 | 0.216 | -0.017 | 0.710 |
| Crop / full ridge | CT-CLIP | 0.254 | 0.266 | 0.244 | 0.237 | -0.051 | 0.266 |

For the original CT with PCA32, COLIPRI led with Uno C 0.634 and IBS 0.201; TANGERINE followed at 0.613 and 0.203. In the crop/full-ridge condition, COLIPRI and TANGERINE reached 0.635 and 0.626 Uno C. Crop/full-ridge improved CT-CLIP discrimination from 0.533 to 0.589 relative to original/full-ridge, but its IBS was 0.237, worse than the cropped PCA32 IBS of 0.212. The [original E1 report](lung1_e1_results_report.md), [crop and ridge results](lung1_e1_crop_ridge_results_report.md), and [crop protocol](lung1_e1_crop_and_ridge_protocol.md) discuss the paired contrasts and their interpretation.

## Interpretation and limits

Within this LUNG1 cohort, the strongest accessible survival signal came from COLIPRI's frozen global representation under the selected-checkpoint protocol. TANGERINE was close on discrimination and never statistically distinguishable from COLIPRI, while Rad-JEPA-3D and CT-CLIP were lower under their original-CT paths and not distinguishable from chance. The 2-D natural-image DINOv3, applied slice by slice and averaged, fell between the two groups. Only its frozen MLP rung was significantly above chance after correction (0.561), and it reached the level of the clinical and radiomics baselines without exceeding them. After Holm correction, no FM condition was significantly better than stage + volume or GTV radiomics, so the advantage of COLIPRI and TANGERINE over the baselines remains suggestive. More trainable capacity did not improve held-out performance at this cohort size: no MLP or SQA head beat its linear probe, and LoRA and partial tuning were significantly worse than the linear probe for TANGERINE and COLIPRI. Different preprocessing and CT-CLIP's different adaptation target limit any causal claim about pretraining objectives or adaptation mechanisms. All evaluations are internal; there is no external validation or clinical-covariate comparison in these reported tables. Calibration requires particular care, and the selected-checkpoint point estimates should not be described as clinically deployable survival predictions.

## E3 follow-up

The [frozen linear Cox label-efficiency experiment](lung1_e3_frozen_linear_report.md) reuses the original-CT embeddings and locked outer test folds with nested 10%, 25%, 50%, and 100% training-label budgets. It reports five-fold mean learning curves, including a fixed eight-component PCA analysis and a 16-component sensitivity. Its fitting and aggregation protocols differ from the pooled main comparison above, so its scores are presented in the separate E3 report.

The [matched E3 baseline comparison](lung1_e3_matched_baselines_report.md) adds stage + tumor volume Cox and GTV radiomics Cox learning curves on the same subsets and outer folds, with paired comparisons against COLIPRI.
