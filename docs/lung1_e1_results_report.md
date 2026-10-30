# Experiment E1 report: prognostic information in frozen CT representations

**Completed:** 2 October 2026  
**Dataset:** TCIA NSCLC-Radiomics (LUNG1)  
**Endpoint:** overall survival, measured from treatment start  
**Primary metric:** Uno's inverse-probability-of-censoring-weighted C-index through 36 months

## Research question and design

Experiment E1 asks how much survival information a low-capacity model can access from each frozen CT foundation model representation. Four pretrained encoders were kept frozen. Each produced one global embedding per patient's selected original CT DICOM series. The analysis used **416 patients**, including **369 deaths** and **47 censored observations**. This is the common, quality-audited cohort eligible for the planned global and GTV analyses; one patient with an eligible global CT was excluded because a GTV mask was unavailable. No outcome labels were used during embedding extraction.

Each embedding was evaluated with the same linear Cox probe: training-fold standardization, 32-component PCA, and L2-regularized Cox regression. The L2 penalty was chosen from 0.01, 0.1, and 1.0 using three inner folds and Uno C. The outer evaluation used five repetitions of five patient-stratified folds, with identical assignments for all encoders. Event status and early versus late stage defined the stratification. Standardization, PCA, and Cox fitting used only the training portion of each fold.

The primary result is the unweighted mean of the 25 held-out fold scores. Secondary endpoints were cumulative/dynamic AUC at 1, 2, and 3 years; the IPCW integrated Brier score (IBS) from 3 to 36 months; and 2-year calibration summaries. The reported 95% intervals come from 200 paired patient bootstrap resamples within the fixed held-out folds. They are **conditional on the fitted models and fold assignments**, and repeated predictions from the same patients limit how broadly they can be interpreted.

## CT input preprocessing

All four encoders started from the **same selected original CT DICOM series for each patient**. SimpleITK read the ordered slices as a three-dimensional HU image. The extraction script checked the DICOM slice count against the cohort audit. It did **not** pass a common resized CT to all models: each encoder received the spatial and intensity transform shown below. Tensor sizes are listed as depth × height × width, excluding batch and channel dimensions.

| Encoder | Spatial preparation | Intensity preparation | Tensor size sent to encoder |
| --- | --- | --- | ---: |
| TANGERINE | Keep the source orientation and full image field of view; B-spline resample to a fixed voxel grid. Output spacing changes with each scan's original size and spacing. | Clip HU to [−1200, 800], then map linearly to [0, 1]. | **256 × 256 × 256** |
| Rad-JEPA-3D | Reorient to RAS; linearly resize the entire array. This fixes voxel counts, not physical millimetre spacing. | Clip HU to [−1000, 400], then map linearly to [0, 1]. | **32 × 256 × 256** |
| COLIPRI | Use the model's TorchIO processor: reorient to RAS, linearly resample to 2 mm isotropic spacing, then centrally crop or pad. The target cube spans about 384 mm per side. | Clamp HU to [−1000, 1000], then map to [−1, 1]; padding uses the minimum intensity. | **192 × 192 × 192** |
| CT-CLIP | Reorient to LPS; trilinearly resize to 0.75 × 0.75 × 1.5 mm spacing (x × y × z), then centrally crop or pad to a roughly 360 mm field of view along each axis. | Clip HU to [−1000, 1000], divide by 1000 to obtain [−1, 1]; padding is −1. | **240 × 480 × 480** |

TANGERINE and Rad-JEPA-3D resize the full CT into fixed voxel counts, while COLIPRI and CT-CLIP use fixed physical spacing followed by a central crop or pad. This can change anatomical coverage across encoders and is part of the model-specific input protocol. No lung mask, GTV mask, or GTV-preserving thorax bounding box was applied in this **original-CT E1 run**. The previously generated thorax crops belong to a separate sensitivity analysis.

Each encoder then produced one frozen global vector: 2,048 features for TANGERINE, 384 for Rad-JEPA-3D, 768 for COLIPRI, and 512 for CT-CLIP. The **32-component PCA was applied to those vectors inside the survival-model training folds**, after CT encoding. It did not make the CT image sizes common, and the reported results measure decodability from those 32-component representations.

[See rendered input slices for LUNG1-001](../outputs/lung1_e1_input_examples/LUNG1-001_comparison.png). Axial, coronal, and sagittal slices were selected through the GTV centroid for display; the GTV mask was not sent to any encoder. The figure comes from the actual model input tensors and can be regenerated with `scripts/render_lung1_e1_input_examples.py`.

## Results

Higher Uno C and AUC indicate better discrimination; lower IBS indicates better prediction error. Every model contributed 25 fold scores and 2,080 held-out predictions (each of the 416 patients appears once in each repetition). No fold metric was missing.

| Frozen encoder | 36-month Uno C (95% bootstrap interval) | AUC 1 year | AUC 2 years | AUC 3 years | IBS 3–36 months | Held-out Cox calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | **0.634** (0.616–0.651) | **0.677** | **0.675** | **0.719** | **0.201** | 1.328 |
| TANGERINE | 0.613 (0.599–0.630) | 0.638 | 0.641 | 0.673 | 0.203 | 1.169 |
| Rad-JEPA-3D | 0.550 (0.534–0.566) | 0.561 | 0.567 | 0.573 | 0.215 | 0.640 |
| CT-CLIP | 0.529 (0.514–0.545) | 0.537 | 0.557 | 0.547 | 0.216 | 0.374 |

At two years, mean predicted minus observed survival was −0.010 for COLIPRI, −0.009 for TANGERINE, −0.006 for Rad-JEPA-3D, and −0.004 for CT-CLIP. These small mean offsets do not establish individual-level calibration.

### Paired differences in the primary metric

Differences below are first model minus second model, using the same patient resamples for both. Positive values favor the first model.

| Comparison | Uno C difference | 95% paired bootstrap interval |
| --- | ---: | ---: |
| COLIPRI − TANGERINE | +0.021 | +0.005 to +0.036 |
| COLIPRI − Rad-JEPA-3D | +0.085 | +0.065 to +0.106 |
| COLIPRI − CT-CLIP | +0.105 | +0.081 to +0.127 |
| TANGERINE − Rad-JEPA-3D | +0.063 | +0.043 to +0.082 |
| TANGERINE − CT-CLIP | +0.084 | +0.064 to +0.106 |
| Rad-JEPA-3D − CT-CLIP | +0.020 | −0.002 to +0.041 |

## Interpretation

On this LUNG1 cohort and with this fixed linear probe, **COLIPRI and TANGERINE exposed more prognostic information than Rad-JEPA-3D and CT-CLIP**. COLIPRI had the highest mean discrimination and lowest mean IBS. The paired interval for Rad-JEPA-3D versus CT-CLIP includes zero, so their ordering is less clear under this analysis. The results address frozen global representation *decodability*; they do not determine how much each model could improve through adaptation, regional pooling, or a different survival head.

These are internal cross-validation results. The intervals do not account for choosing this cohort, encoder implementation, or probe design; they are not adjusted for six pairwise comparisons. External validation and clinical or tumor-volume baselines are still needed before interpreting the ranking as a general result for NSCLC prognosis. This run used each encoder's preprocessing on the **original CT**, not the separate GTV-preserving thorax crop sensitivity input.

The CT-CLIP checkpoint came from a public mirror because the authors' CT-RATE repository required Hugging Face access in this environment. Its identity with the gated original has not been verified. CT-CLIP comparisons should therefore retain that provenance qualification.

## Reproducibility and source artifacts

- [Full protocol and input hashes](../outputs/lung1_e1_frozen_cox_four_models/protocol.json)
- [Machine-readable summary](../outputs/lung1_e1_frozen_cox_four_models/summary.json)
- [Per-fold metrics](../outputs/lung1_e1_frozen_cox_four_models/fold_metrics.csv)
- [Held-out patient predictions](../outputs/lung1_e1_frozen_cox_four_models/out_of_fold_predictions.csv)
- [Locked outer folds](../configs/lung1_e1_folds.csv) and [inner fold assignments](../outputs/lung1_e1_frozen_cox_four_models/inner_folds.csv)
- [Experiment implementation and extraction details](lung1_e1_frozen_cox.md), [cohort audit](lung1_cohort_audit.md), and [CT-CLIP checkpoint metadata](../external/ctclip/checkpoint_metadata.json)

The encoder inference used GPUs; Cox fitting and metrics ran on the CPU. The frozen embedding manifests and checkpoint hashes are recorded in the protocol and each model's extraction provenance under `outputs/lung1_e1_embeddings/`.
