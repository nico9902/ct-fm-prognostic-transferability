# LUNG1 E4: frozen-token regional linear Cox probes

**Run date:** 9 October 2026  
**Cohort:** 416 LUNG1 patients with GTV, original CT frozen-token caches  
**Models:** COLIPRI, TANGERINE, Rad-JEPA-3D, CT-CLIP

## Question and design

Where are useful prognostic **token positions** in each frozen representation? We compared five fixed pooling rules using the same cached final-layer tokens, five locked E3 outer folds (repetition zero), training-only standardization, 16-component PCA, and L2-regularized linear Cox. Each outer development pool contains 332–333 patients; the outer test pool has 83–84. Penalty selection uses three event-stratified inner folds and the existing 0.01, 0.1, 1.0 grid. PCA and the Cox model are refitted inside every inner fold and on the outer development pool. There are 100 primary fits: four models × five regions × five folds. No GPU was used for pooling or Cox fitting.

The primary metric is the mean of five held-out **Uno C-indices through 36 months**. Full-follow-up Harrell C and Uno C, time-dependent AUC, and integrated Brier score are secondary. For outer evaluation, the censoring curve is estimated from the corresponding outer development patients and does not select the penalty. The same patient and fold assignments are used for every region.

| Region | One patient-level vector from saved tokens |
|---|---|
| All | Mean of every saved token. This is the within-E4 reference. |
| Tumor | Mean weighted by each block's physical GTV volume fraction. |
| Non-GTV | Mean of blocks with exactly zero physical GTV footprint. |
| Peri20 | Mean weighted by the physical GTV shell at 0–20 mm. |
| Tumor + peri20 | Mean weighted by the union of the GTV and its 0–20 mm shell. |

Every weighted mean is normalized by its sum of weights. The union fraction is capped at one where 1 mm voxel-center discretization makes the geometric estimate slightly exceed a block volume. The same 16-dimensional PCA and Cox capacity is applied to every resulting vector. These features are **not** the E3 global embedding; the all-token mean is the matched reference for this experiment.

**Why COLIPRI E3 full-budget is 0.617 while E4 all-token is 0.591.** E3 uses COLIPRI's official `pool=True` learned attention-pooled embedding; E4 all-token uses a uniform mean of dense spatial tokens extracted with `pool=False` and stored on the 8³ grid. The same encoder checkpoint, original-CT transform, cohort, locked folds, 16-component PCA, Cox fitting and Uno C evaluation are used. Feeding the E3 global embeddings through the E4 Cox fitting function reproduced the E3 fold scores exactly (mean 0.616705), so the gap is due to the feature representation rather than the survival-probe implementation. For COLIPRI's 24³-to-8³ equal-block averaging, the uniform global mean is preserved apart from float16 cache rounding; the key pooling difference is learned attention versus uniform averaging. The exact numerical contribution of attention versus cache rounding was not separately measured.

GTV and shell geometry are outcome-independent. The shell was generated from the cached 1 mm GTV with a Euclidean distance transform, extending beyond the cached CT crop where needed; ring points were limited to the original CT field and each model's input field. See the [GTV mapping audit](lung1_e4_gtv_token_mapping_report.md), [GTV mapping code](../scripts/audit_lung1_e4_gtv_token_mapping.py), and [shell mapping code](../scripts/audit_lung1_e4_peritumor_mapping.py).

## Mapping quality and spatial resolution

Every patient has a nonempty tumor and peri20 pool for all four models. Median tumor-overlapping saved-token counts are 7, 6, 4, and 8 for COLIPRI, TANGERINE, Rad-JEPA-3D, and CT-CLIP; median peri20-overlapping counts are 18, 14, 9.5, and 20, respectively. Tumor and shell blocks overlap frequently. The number of patients with **no shell-only block** is 10, 7, 20, and 10, respectively. Thus the peri20 probe is not anatomically independent of the tumor probe.

For LUNG1-038, COLIPRI retains 77.3% of the shell and CT-CLIP retains 67.1% in their input fields. The primary GTV audit found 93.1% and 79.4% tumor retention for this patient. The sensitivity analysis below removes this case and the two Rad-JEPA cases whose GTV vanishes under nearest-neighbor sampling of its 32-slice input.

The saved nominal blocks span about 45–95 mm on each axis, much wider than the proposed 0–5, 5–10, and 10–20 mm rings. This experiment uses one pooled 0–20 mm shell; it does not claim to resolve individual narrow rings.

## Primary 16-component results

Mean held-out Uno C through 36 months:

| Model | All | Tumor | Non-GTV | Peri20 | Tumor + peri20 | Tumor − all (paired 95% interval) |
|---|---:|---:|---:|---:|---:|---:|
| COLIPRI | 0.591 | 0.597 | 0.589 | 0.604 | 0.605 | +0.006 [−0.019, +0.031] |
| TANGERINE | 0.582 | 0.573 | 0.574 | 0.593 | 0.590 | −0.010 [−0.054, +0.035] |
| Rad-JEPA-3D | 0.539 | 0.472 | 0.544 | 0.491 | 0.489 | −0.067 [−0.110, −0.025] |
| CT-CLIP | 0.526 | 0.561 | 0.518 | 0.547 | 0.553 | +0.035 [−0.015, +0.081] |

Mean full-follow-up Harrell C:

| Model | All | Tumor | Non-GTV | Peri20 | Tumor + peri20 |
|---|---:|---:|---:|---:|---:|
| COLIPRI | 0.587 | 0.594 | 0.585 | 0.604 | 0.605 |
| TANGERINE | 0.582 | 0.574 | 0.574 | 0.593 | 0.590 |
| Rad-JEPA-3D | 0.540 | 0.478 | 0.544 | 0.494 | 0.493 |
| CT-CLIP | 0.521 | 0.559 | 0.513 | 0.539 | 0.545 |

The [regional score plot](../outputs/lung1_e4_regional_cox/regional_uno_c.png) shows conditional 95% patient-bootstrap intervals. [All paired regional differences](../outputs/lung1_e4_regional_cox/paired_regional_differences.csv) and [between-model difference-of-differences](../outputs/lung1_e4_regional_cox/between_model_regional_interactions.csv) are saved. The bootstrap resamples held-out patients within each fixed fold and shares resamples across models and regions; it does not refit models, represent a new cohort, or adjust for multiple comparisons.

The clearest within-model result is that Rad-JEPA-3D loses discrimination when restricted to tumor-weighted positions. COLIPRI changes little across regional pools. CT-CLIP's tumor pool is numerically better than its all-token pool, but its paired interval includes zero. Non-GTV and all-token performance is close for every model; the GTV touches only a small fraction of the 256 or 512 saved blocks, so these two averages use mostly the same positions.

The regional profiles differ descriptively between encoders. For tumor-minus-all, COLIPRI minus Rad-JEPA-3D is +0.074 (conditional 95% interval +0.023 to +0.121), and CT-CLIP minus Rad-JEPA-3D is +0.102 (+0.046 to +0.158). These exploratory interactions are sensitive to representation, pooling and probe capacity; the intervals are not multiplicity adjusted.

## Sensitivity analyses

**Eight PCA components.** We repeated all 100 fits with only the PCA dimension changed. Tumor-minus-all is +0.021 for COLIPRI, +0.015 for TANGERINE, −0.055 for Rad-JEPA-3D, and +0.003 for CT-CLIP. Rad-JEPA's tumor drop persists (conditional interval −0.103 to −0.010); CT-CLIP's apparent tumor gain nearly disappears. [Complete PCA8 results](../outputs/lung1_e4_regional_cox_pca8/summary.csv).

**Geometry edge cases removed.** We removed LUNG1-038, LUNG1-108, and LUNG1-192 from both fitting and testing, retained their original fold assignments for everyone else, and repeated all 100 16-component fits on 413 patients. Tumor-minus-all is +0.005, −0.012, −0.066, and +0.021 for COLIPRI, TANGERINE, Rad-JEPA-3D, and CT-CLIP, respectively. Rad-JEPA's contrast remains similar; CT-CLIP's positive contrast is smaller. [Complete exclusion results](../outputs/lung1_e4_regional_cox_exclude3/summary.csv).

## Interpretation and limits

These are frozen final-layer token probes. A token at a tumor-overlapping position can contain information gathered from other positions by the encoder; the tumor and peri20 pools can use some of the same tokens. Regional differences therefore locate **decodable information by token position**, not its exclusive tissue source. In particular, the weaker Rad-JEPA tumor probe does not show that its tumor tissue contains no prognostic information.

The GTV segmentation is required to construct and use the tumor, non-GTV, and peri20 features. E3's original-CT global embedding did not require GTV at inference, so direct score comparison between E3 and these E4 probes includes that input difference as well as different pooling. This is internal cross-validation on a cohort previously used to develop E1–E3 analyses, not external validation. The 16-component choice was informed by E3 and these regional comparisons remain exploratory.

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  .venv/bin/python scripts/audit_lung1_e4_gtv_token_mapping.py --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  .venv/bin/python scripts/audit_lung1_e4_peritumor_mapping.py --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e4_regional_cox.py
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_regional_cox.py --bootstrap 300
```

The Cox script was also run with `--components 8 --output-dir outputs/lung1_e4_regional_cox_pca8`, and separately with `--exclude-patients LUNG1-038,LUNG1-108,LUNG1-192 --output-dir outputs/lung1_e4_regional_cox_exclude3`; the analysis script was then applied to each output directory. Each directory contains its locked protocol, 100 individual fit records, fold-level metrics, held-out predictions, regional summaries, paired intervals, and a plot. All 300 primary Uno C and Harrell C fold scores are finite; all mapping arrays have the expected shape and finite values.

## Follow-up: GTV filling before the frozen encoder

**Run date:** 10 October 2026

This follow-up asks whether changing the **input tumor voxels** changes the prognostic global embedding. It differs from the regional-token probes above: the volume is altered before encoding, and the official E1 global embedding is used with a 16-component linear Cox probe. For each patient and model we re-encode the cached original input, the same input with the GTV filled, and, where possible, a matched same-shape control patch filled. The original re-encoding provides a within-run reference and is checked against E3.

The GTV is resampled from its cached segmentation to each model's input grid. The intervention is applied **after the model's geometric and intensity preprocessing, immediately before its frozen encoder**. The GTV core is replaced by the median input intensity in a local 5–20 mm exterior shell, with a 3 mm feather outside the mask. The same procedure is applied to the control patch. The patch translates the exact source-CT GTV shape within the same axial levels, avoids tumor overlap, requires at least 95% of its voxels above −950 HU and at least 30% overlap with the cached lung segmentation, and minimizes a prespecified image-intensity and distance score. Outcomes are not used to place it. A valid control is available for 403 of 416 patients; the other 13 are excluded from **all three conditions** in the control comparison.

Both comparisons preserve the locked E3 five outer folds and refit the training-only standardization, 16-component PCA, and ridge Cox model separately for every condition. Penalty selection uses the same three event-stratified inner folds and 0.01, 0.1, 1.0 grid. The full-set comparison has 416 patients (369 observed deaths, 286 by 36 months); the matched control comparison has 403 (357 observed deaths, 274 by 36 months). The latter outer development pools contain 320–324 patients and the held-out pools contain 79–83. The primary metric is the mean held-out Uno C through 36 months; full-follow-up Harrell C is secondary. Paired intervals resample held-out patients within each fixed fold, without refitting.

### Results on all 416 patients

| Model | Original Uno 36m | GTV-filled Uno 36m | Original Harrell full | GTV-filled Harrell full | Original − GTV Uno (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| COLIPRI | 0.617 | 0.599 | 0.614 | 0.599 | +0.0179 [+0.0021, +0.0354] |
| TANGERINE | 0.599 | 0.602 | 0.595 | 0.597 | −0.0034 [−0.0066, −0.00001] |
| Rad-JEPA-3D | 0.539 | 0.540 | 0.540 | 0.540 | −0.0012 [−0.0025, +0.00001] |
| CT-CLIP | 0.528 | 0.529 | 0.528 | 0.528 | −0.0006 [−0.0054, +0.0042] |

The re-encoded original Uno 36-month score differs from the E3 full-budget score by +0.00006 for COLIPRI, +0.00001 for TANGERINE, exactly zero for Rad-JEPA-3D, and −0.00152 for CT-CLIP. The comparison uses the re-encoded original for every masked-volume contrast. CT-CLIP's small baseline drift is consistent with the float16 cached input, but its exact source was not isolated.

### Same-patient, same-shape control comparison

All three conditions below are **refit and evaluated on the same 403 patients**. Each score cell shows Uno C through 36 months / full-follow-up Harrell C. The last column is the control-minus-GTV contrast: positive values mean a larger score after the control fill than after the GTV fill.

| Model | Original | GTV filled | Control filled | Original − GTV Uno (paired 95% interval) | Control − GTV Uno (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| COLIPRI | 0.623 / 0.619 | 0.604 / 0.603 | 0.616 / 0.612 | +0.0189 [+0.0033, +0.0318] | +0.0124 [−0.0060, +0.0311] |
| TANGERINE | 0.594 / 0.590 | 0.595 / 0.591 | 0.600 / 0.595 | −0.0009 [−0.0033, +0.0019] | +0.0052 [+0.0003, +0.0111] |
| Rad-JEPA-3D | 0.544 / 0.545 | 0.545 / 0.545 | 0.545 / 0.545 | −0.0001 [−0.0027, +0.0021] | −0.00002 [−0.0017, +0.0017] |
| CT-CLIP | 0.517 / 0.518 | 0.518 / 0.519 | 0.516 / 0.517 | −0.0007 [−0.0057, +0.0044] | −0.0018 [−0.0092, +0.0053] |

COLIPRI is the clearest response to the GTV intervention. Its original-to-GTV score drop persists in the 403-patient subset. The control fill also lowers its score by 0.0066, and the direct **control-minus-GTV** interval includes zero. Thus these results do not establish that the extra COLIPRI drop is specific to tumor tissue. For TANGERINE, the small positive control-minus-GTV contrast comes from the control score rising above the original score; the original-to-GTV contrast on the same patients is nearly zero. Rad-JEPA-3D and CT-CLIP show little score response to either fill.

The input perturbations reached the encoders. Median relative L2 change in the global embedding on the 403 matched patients is 0.331 after GTV fill versus 0.144 after control fill for COLIPRI; 0.0012 versus 0.0015 for TANGERINE; 0.0009 versus 0.0010 for Rad-JEPA-3D; and 0.0153 versus 0.0164 for CT-CLIP. Median GTV-to-control ratios of mean absolute *input* intensity change per filled voxel are 1.03, 1.03, 1.09, and 0.97, respectively. COLIPRI's global embedding is therefore much more responsive to the GTV fill than to the control fill, even though this did not yield a clearly separated control-minus-GTV survival score. Two small Rad-JEPA GTV masks and two control masks vanish at its 32-slice input resolution; all other masks are nonempty after model input sampling.

### Post-hoc COLIPRI fill-method sensitivity

Because COLIPRI was the only model with a substantial score decrease under shell-median filling, we selected it **after the primary comparison** for a second fill method. For each GTV or control voxel, this method copies the nearest exterior input intensity, smooths the resulting field with a 2 mm Gaussian kernel, and uses the same 3 mm exterior feather. The [visual check](../outputs/lung1_e4_masked_volume/qc/LUNG1-001_colipri_inpaint.png) shows that this avoids the uniform core made by the shell-median fill. The original global embedding, patients, controls, locked folds, PCA16, and Cox procedure remain the same.

| Patients | Original Uno 36m / Harrell full | GTV inpaint Uno 36m / Harrell full | Control inpaint Uno 36m / Harrell full | Original − GTV Uno (paired 95% interval) | Control − GTV Uno (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| All 416 | 0.617 / 0.614 | 0.614 / 0.613 | — | +0.0027 [−0.0115, +0.0183] | — |
| Matched 403 | 0.623 / 0.619 | 0.621 / 0.620 | 0.622 / 0.618 | +0.0014 [−0.0136, +0.0146] | +0.0005 [−0.0155, +0.0138] |

The 0.0179 shell-median drop on all 416 patients shrinks to 0.0027 with this inpainting method. On the matched 403 patients, the direct control-minus-GTV difference is nearly zero. This makes the apparent prognostic drop **sensitive to how the missing region is filled**. The GTV inpainting still changes COLIPRI's global embedding substantially (median relative L2 0.357 on the matched patients), while the inpainted control changes it by only 0.015. The median GTV-to-control ratio of mean absolute input change per mask voxel is 2.01, so the inpainted control is a weaker intervention. Inpainting may also preserve mass-like boundary information. This post-hoc sensitivity supports caution about the shell-median result; it does not establish the true effect of removing tumor information.

### Black GTV sensitivity (12 October 2026)

We also tested a hard black mask for **all four frozen encoders**. After each model's usual geometric and intensity preprocessing and immediately before the encoder, only the GTV input voxels were set to that model's minimum input value: **0** for TANGERINE and Rad-JEPA-3D, **−1** for COLIPRI and CT-CLIP. There is no feathering or smoothing. The same operation was applied to the previously selected same-shape control patch. The [COLIPRI visual check](../outputs/lung1_e4_masked_volume/qc/LUNG1-001_colipri_black.png) shows the GTV and control placements. Original embeddings, PCA16 Cox fitting, locked folds, and evaluation are as above; every model has 416 GTV-black and 403 control-black vectors.

Mean held-out Uno C through 36 months and full-follow-up Harrell C on all 416 patients:

| Model | Original Uno / Harrell | GTV black Uno / Harrell | Original − GTV Uno (paired 95% interval) |
|---|---:|---:|---:|
| COLIPRI | 0.617 / 0.614 | 0.593 / 0.593 | +0.0234 [−0.0013, +0.0450] |
| TANGERINE | 0.599 / 0.595 | 0.607 / 0.602 | −0.0081 [−0.0138, −0.0020] |
| Rad-JEPA-3D | 0.539 / 0.540 | 0.541 / 0.541 | −0.0019 [−0.0062, +0.0019] |
| CT-CLIP | 0.528 / 0.528 | 0.526 / 0.525 | +0.0019 [−0.0064, +0.0094] |

The matched-control comparison refits and evaluates all three conditions on the same 403 patients. Each score cell is Uno C through 36 months / full-follow-up Harrell C:

| Model | Original | GTV black | Control black | Original − GTV Uno (paired 95% interval) | Control − GTV Uno (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| COLIPRI | 0.623 / 0.619 | 0.600 / 0.599 | 0.629 / 0.624 | +0.0226 [+0.0016, +0.0448] | +0.0287 [+0.0057, +0.0512] |
| TANGERINE | 0.594 / 0.590 | 0.602 / 0.598 | 0.613 / 0.607 | −0.0083 [−0.0147, −0.0020] | +0.0104 [+0.0027, +0.0178] |
| Rad-JEPA-3D | 0.544 / 0.545 | 0.547 / 0.547 | 0.546 / 0.546 | −0.0023 [−0.0059, +0.0015] | −0.0004 [−0.0030, +0.0022] |
| CT-CLIP | 0.517 / 0.518 | 0.518 / 0.517 | 0.518 / 0.520 | −0.0002 [−0.0096, +0.0085] | +0.0005 [−0.0086, +0.0095] |

COLIPRI's black-GTV contrast separates from its same-shape black control under this fixed hard-mask procedure, but its nearly unchanged score under inpainting shows that the result is **not stable across replacement methods**. The black control improves COLIPRI's score above its original score. TANGERINE likewise improves under both black interventions, more with the control. These are signs that the hard patch itself and its location affect discrimination. Median relative L2 changes in COLIPRI's global vector are 0.545 for GTV black and 0.296 for control black on the matched patients. Median ratios of GTV-to-control mean absolute *input* change per mask voxel are 1.23, 1.19, 1.26, and 1.23 for COLIPRI, TANGERINE, Rad-JEPA-3D, and CT-CLIP, respectively. Thus the GTV and control black patches are shape matched but not equal-strength perturbations. Two Rad-JEPA GTV masks and two control masks remain empty at its input resolution. The black experiment supports a **location-sensitive COLIPRI response to a hard blackout**, not a fill-independent causal estimate of tumor prognostic information.

### Interpretation and artifacts

These are exploratory internal cross-validation results, conditional on the fixed patients, folds, masks, fill methods, and fitted Cox models. The paired 95% intervals use 300 held-out patient resamples per outer fold; they do not include uncertainty from refitting, external cohorts, choosing the fill method, or multiple comparisons. A local median fill creates a synthetic, flatter region; a black mask creates an even more conspicuous patch. The control patch is same-shape but not perfectly tissue matched: its median lung-mask overlap is 39%, and its median absolute GTV-to-control intensity gap is 56 HU. The 13 patients without a valid control were removed before *all* control-comparison fits, so the 403-patient scores should not be compared as if they were on the same evaluation set as the 416-patient scores. Only COLIPRI received the inpainting method, selected after its primary result. External validation and stronger tissue-matched controls are needed before attributing a change in prognostic discrimination to tumor tissue.

The [full-set scores](../outputs/lung1_e4_masked_volume/cox_full416/summary.csv), [matched-control scores](../outputs/lung1_e4_masked_volume/cox_control403/summary.csv), [paired differences](../outputs/lung1_e4_masked_volume/cox_control403/paired_mask_differences.csv), [embedding-change summary](../outputs/lung1_e4_masked_volume/cox_control403/embedding_change_summary.csv), [input-change summary](../outputs/lung1_e4_masked_volume/cox_control403/input_change_summary.csv), and [matched-control plot](../outputs/lung1_e4_masked_volume/cox_control403/masked_volume_uno_c.png) are saved. The complete [control manifest](../outputs/lung1_e4_masked_volume/control_masks/control_manifest.csv) identifies the 13 missing controls. All four extraction directories contain 416 original, 416 GTV-filled, and 403 control-filled frozen embeddings, with per-patient provenance and completion markers.

The post-hoc COLIPRI [full-set inpainting scores](../outputs/lung1_e4_masked_volume/cox_inpaint_full416/summary.csv), [matched inpainting scores](../outputs/lung1_e4_masked_volume/cox_inpaint_control403/summary.csv), [paired differences](../outputs/lung1_e4_masked_volume/cox_inpaint_control403/paired_mask_differences.csv), [embedding changes](../outputs/lung1_e4_masked_volume/cox_inpaint_control403/embedding_change_summary.csv), and [matched plot](../outputs/lung1_e4_masked_volume/cox_inpaint_control403/masked_volume_uno_c.png) are also saved.

The black-mask [full-set scores](../outputs/lung1_e4_masked_volume/cox_black_full416/summary.csv), [full-set paired differences](../outputs/lung1_e4_masked_volume/cox_black_full416/paired_mask_differences.csv), [matched-control scores](../outputs/lung1_e4_masked_volume/cox_black_control403/summary.csv), [matched paired differences](../outputs/lung1_e4_masked_volume/cox_black_control403/paired_mask_differences.csv), [input-change summary](../outputs/lung1_e4_masked_volume/cox_black_control403/input_change_summary.csv), [embedding-change summary](../outputs/lung1_e4_masked_volume/cox_black_control403/embedding_change_summary.csv), and [matched plot](../outputs/lung1_e4_masked_volume/cox_black_control403/masked_volume_uno_c.png) are saved separately.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  .venv/bin/python scripts/prepare_lung1_e4_masked_volume_controls.py --workers 4
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  .venv/bin/python scripts/extract_lung1_e4_masked_global_embeddings.py --model colipri --device cuda:0
# Repeat extraction for tangerine, radjepa, ctclip, selecting an available GPU.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e4_masked_global_cox.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e4_masked_global_cox.py --include-sham
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_masked_global_cox.py --bootstrap 300
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_masked_global_cox.py \
    --output-dir outputs/lung1_e4_masked_volume/cox_control403 --bootstrap 300
# Post-hoc COLIPRI sensitivity.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  .venv/bin/python scripts/extract_lung1_e4_masked_global_embeddings.py \
    --model colipri --device cuda:0 --fill-method inpaint
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_lung1_e4_masked_global_cox.py --fill-method inpaint
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_lung1_e4_masked_global_cox.py --fill-method inpaint --include-sham
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_masked_global_cox.py \
    --output-dir outputs/lung1_e4_masked_volume/cox_inpaint_full416 --bootstrap 300
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_masked_global_cox.py \
    --output-dir outputs/lung1_e4_masked_volume/cox_inpaint_control403 --bootstrap 300
# Black GTV and control: repeat extraction for each of the four models.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  .venv/bin/python scripts/extract_lung1_e4_masked_global_embeddings.py \
    --model colipri --device cuda:0 --fill-method black
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_lung1_e4_masked_global_cox.py --fill-method black
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_lung1_e4_masked_global_cox.py --fill-method black --include-sham
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_masked_global_cox.py \
    --output-dir outputs/lung1_e4_masked_volume/cox_black_full416 --bootstrap 300
MPLCONFIGDIR=/tmp/lung1_e4_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e4_masked_global_cox.py \
    --output-dir outputs/lung1_e4_masked_volume/cox_black_control403 --bootstrap 300
```

## Later E4 additions (17 October 2026)

- DINOv3 regional probes: [DINOv3 E3/E4 report](lung1_dinov3_e3_e4_report.md).
- GTV tokens hidden inside all five frozen encoders (attention masking, no volume fill): [token-masking report](lung1_e4_token_masking_report.md).
