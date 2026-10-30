# E1 follow-up report: thorax crop and full-embedding Cox probes

**Completed:** 2 October 2026  
**Cohort:** LUNG1, 416 patients, 369 deaths and 47 censored observations  
**Endpoint:** overall survival from treatment start  
**Primary measure:** held-out Uno C-index through 36 months (higher is better)

## Summary

The strongest cropped-CT full-embedding result was **COLIPRI, Uno C 0.635**, followed by TANGERINE 0.626, CT-CLIP 0.589 and Rad-JEPA-3D 0.576. These are means across 25 held-out folds; the same patients and outer folds were used in all four conditions.

At a fixed full-embedding ridge probe, cropping increased Uno C by **0.056 for CT-CLIP** and **0.046 for Rad-JEPA-3D** relative to original CTs. The paired, patient-cluster bootstrap intervals were +0.017 to +0.099 and +0.009 to +0.085, respectively. COLIPRI changed by +0.003 and TANGERINE by −0.008, with intervals spanning zero. The crop therefore did not improve every encoder.

Full-embedding ridge improved discrimination over PCA32 for TANGERINE on original CTs (+0.021). On cropped CTs, the ridge-minus-PCA32 differences were small and their patient-cluster intervals all spanned zero. **A higher C-index did not always improve survival probability accuracy:** cropped CT-CLIP ridge had Uno C 0.589 but an integrated Brier score (IBS) of 0.237, versus Uno C 0.578 and IBS 0.212 with cropped PCA32.

## Inputs and evaluation

The original-CT condition reads the selected DICOM series. The crop condition reads a 1 mm isotropic, float32 HU bounding box around both lungs and the GTV with a 10 mm margin. All 416 stored crops retain 100% of the GTV. They remain in HU: intensity clipping and normalization are applied **once by each encoder's input path**, after reading the crop. The same model-specific transforms are used for original and cropped CTs.

| Encoder | Encoder input `(D,H,W)` | HU window and normalized range | Global embedding sent to full ridge Cox |
| --- | ---: | --- | ---: |
| TANGERINE | 256 × 256 × 256 | [−1200, 800] → [0, 1] | 2,048 |
| Rad-JEPA-3D | 32 × 256 × 256 | [−1000, 400] → [0, 1] | 384 |
| COLIPRI | 192 × 192 × 192 | [−1000, 1000] → [−1, 1] | 768 |
| CT-CLIP | 240 × 480 × 480 | [−1000, 1000] → [−1, 1] | 512 |

The [LUNG1-001 input comparison](../outputs/lung1_e1_input_examples/thorax_hu/LUNG1-001_comparison.png) shows slices from the exact normalized tensors immediately before inference; its [audit JSON](../outputs/lung1_e1_input_examples/thorax_hu/LUNG1-001_input_audit.json) records tensor shapes and ranges. The [input protocol](lung1_e1_crop_and_ridge_protocol.md) gives the orientations, resampling, padding and crop details. The GTV mask selects the example's display slices; no GTV mask is passed to an encoder.

Each of the four conditions uses **five repetitions of five outer patient-stratified folds** (25 held-out evaluations per encoder). Every patient has one held-out prediction per repetition. The outer repetition seeds are 20261001–20261005; three inner folds select the Cox L2 penalty. Standardization and PCA, when used, are fitted within the training folds. No outcome label is used during embedding extraction. The PCA32 probe reduces each standardized vector to 32 components and tunes the penalty over {0.01, 0.1, 1}. The full-embedding ridge probe retains all coordinates and tunes over {0.01, 0.1, 1, 10}. It fits an Efron-tie Cox partial likelihood with L-BFGS and uses a Breslow baseline for predicted survival curves. The full-embedding penalty scaling and optimizer differ from the PCA32 implementation, so a probe contrast measures the *whole fitted probe protocol*, not PCA alone.

All intervals in this report were recalculated using **500 paired patient-cluster bootstrap resamples** shared across conditions and repetitions. They are percentile intervals conditional on the trained fold models and locked folds; models were not refitted in each resample. This method produces wider intervals than the 200 within-fold bootstrap intervals in the original E1 report because it preserves repeated appearances of the same patient across repetitions. Contrasts are exploratory and were not adjusted for multiple comparisons.

## Held-out discrimination

Each cell gives mean 36-month Uno C across the 25 outer folds, with a 95% patient-cluster bootstrap interval.

| Encoder | Original CT + PCA32 | Original CT + full ridge | Thorax crop + PCA32 | Thorax crop + full ridge |
| --- | ---: | ---: | ---: | ---: |
| COLIPRI | 0.634 (0.602–0.670) | 0.633 (0.602–0.667) | 0.631 (0.600–0.666) | **0.635 (0.603–0.670)** |
| TANGERINE | 0.613 (0.584–0.644) | 0.634 (0.600–0.663) | 0.616 (0.585–0.647) | 0.626 (0.589–0.655) |
| Rad-JEPA-3D | 0.550 (0.518–0.582) | 0.530 (0.499–0.558) | 0.564 (0.530–0.599) | 0.576 (0.544–0.609) |
| CT-CLIP | 0.529 (0.497–0.561) | 0.533 (0.502–0.562) | 0.578 (0.549–0.608) | 0.589 (0.558–0.622) |

### Paired crop effects

Differences are crop minus original CT at a fixed Cox probe. Positive values favor the crop. Brackets give paired 95% patient-cluster bootstrap intervals.

| Encoder | PCA32 crop effect | Full ridge crop effect |
| --- | ---: | ---: |
| COLIPRI | −0.004 [−0.017, +0.010] | +0.003 [−0.008, +0.015] |
| TANGERINE | +0.003 [−0.016, +0.022] | −0.008 [−0.030, +0.013] |
| Rad-JEPA-3D | +0.015 [−0.024, +0.052] | **+0.046 [+0.009, +0.085]** |
| CT-CLIP | **+0.049 [+0.008, +0.090]** | **+0.056 [+0.017, +0.099]** |

### Paired full-ridge effects

Differences are full ridge minus PCA32 at a fixed CT source. Positive values favor full ridge for *discrimination*.

| Encoder | Original CT ridge effect | Crop ridge effect |
| --- | ---: | ---: |
| COLIPRI | −0.002 [−0.013, +0.008] | +0.005 [−0.005, +0.015] |
| TANGERINE | **+0.021 [+0.002, +0.043]** | +0.009 [−0.003, +0.023] |
| Rad-JEPA-3D | **−0.020 [−0.036, −0.004]** | +0.011 [−0.003, +0.025] |
| CT-CLIP | +0.004 [−0.016, +0.024] | +0.011 [−0.015, +0.037] |

For the crop full-ridge condition, COLIPRI versus TANGERINE differed by only +0.010 Uno C; that run's within-fold paired interval for this encoder comparison included zero. CT-CLIP versus Rad-JEPA-3D was likewise close (+0.014), with an interval that included zero. The ranking within those pairs should not be treated as settled.

## Survival probability performance

The crop full-ridge run also produced held-out cumulative/dynamic AUC, IBS over 3–36 months, and two-year calibration summaries. AUC is higher-is-better; IBS is lower-is-better. The two-year offset is mean predicted minus Kaplan–Meier observed survival. A calibration slope closer to 1 is preferable.

| Encoder | AUC 1 y | AUC 2 y | AUC 3 y | IBS 3–36 m | 2 y offset | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | 0.673 | 0.675 | 0.718 | **0.200** | −0.011 | 0.743 |
| TANGERINE | 0.665 | 0.657 | 0.688 | 0.203 | −0.028 | 0.736 |
| Rad-JEPA-3D | 0.619 | 0.602 | 0.599 | 0.216 | −0.017 | 0.710 |
| CT-CLIP | 0.615 | 0.623 | 0.613 | 0.237 | −0.051 | 0.266 |

| Encoder | Original CT + PCA32 IBS | Original CT + ridge IBS | Crop + PCA32 IBS | Crop + ridge IBS |
| --- | ---: | ---: | ---: | ---: |
| COLIPRI | 0.201 | 0.200 | 0.201 | 0.200 |
| TANGERINE | 0.203 | 0.212 | 0.202 | 0.203 |
| Rad-JEPA-3D | 0.215 | 0.231 | 0.214 | 0.216 |
| CT-CLIP | 0.216 | 0.242 | **0.212** | 0.237 |

The cropped CT-CLIP ridge model ranks patients better than cropped CT-CLIP PCA32, yet its survival probabilities are worse by IBS and its calibration slope is 0.266. It should therefore not be presented as an unqualified improvement. COLIPRI has the lowest or tied-lowest IBS across the four conditions.

## Interpretation and limitations

The crop appears useful for exposing prognostic information to Rad-JEPA-3D and CT-CLIP, while COLIPRI is stable across input source and Cox probe. This is an association in an internally validated cohort, not evidence that the crop improves external prognosis. The bounding box includes the annotated GTV when needed, so the crop field of view uses information unavailable in a purely annotation-free deployment. The **stored crop** retains the full GTV, but the model-specific fixed-size center crop or pad was not audited for GTV retention in every final tensor.

The full ridge probe preserves all embedding coordinates, but correlated high-dimensional features and only about 333 patients per outer training fold make penalty choice important. Changes between PCA32 and ridge cannot be attributed solely to retaining low-variance directions because the implementations and penalty grids differ. Clinical covariates, tumor-volume baselines, external validation and calibration improvement remain necessary before clinical interpretation. The CT-CLIP checkpoint came from a public mirror and has not been verified against the authors' gated original.

## Reproducibility artifacts

- [Paired condition estimates and intervals](../outputs/lung1_e1_condition_comparison/paired_uno_c.json); [comparison script](../scripts/compare_lung1_e1_conditions.py).
- Original CT: [PCA32 summary](../outputs/lung1_e1_frozen_cox_four_models/summary.json), [full ridge summary](../outputs/lung1_e1_frozen_cox_original_ridge/summary.json).
- Thorax crop: [PCA32 summary](../outputs/lung1_e1_frozen_cox_crop_pca32/summary.json), [full ridge summary](../outputs/lung1_e1_frozen_cox_crop_ridge/summary.json).
- Each run directory also contains `protocol.json`, `fold_metrics.csv`, `out_of_fold_predictions.csv` and `inner_folds.csv`. The [locked outer folds](../configs/lung1_e1_folds.csv) are shared by all four runs.
- [Original E1 report](lung1_e1_results_report.md), [crop and ridge input protocol](lung1_e1_crop_and_ridge_protocol.md), [cohort audit](lung1_cohort_audit.md), and [CT-CLIP checkpoint metadata](../external/ctclip/checkpoint_metadata.json).
