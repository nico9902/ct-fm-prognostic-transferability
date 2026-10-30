# LUNG1 E1/E2: literature baselines on the selected 5-fold split

**Run date:** 15 October 2026  
**Protocol:** [one five-fold CV with validation selection](lung1_bf16_selected_protocol.md)

## Setup

The two E3 literature baselines were refitted on the exact E1/E2 selected-checkpoint split. The cohort is the same 416 LUNG1 patients with 369 deaths. Repetition zero of `configs/lung1_e1_folds.csv` gives the five outer test folds. Inner fold zero of each outer training set is the validation set. Each fold has 221–222 training, 111 validation and 83–84 test patients.

- **Stage + tumor volume:** stage II, IIIa and IIIb indicators (stage I reference), a missing-stage indicator, and log(1 + GTV volume in cm³).
- **GTV radiomics:** the 107 PyRadiomics 3.0.1 `Original` features already extracted for E3 (`outputs/lung1_e3_baselines/radiomics_features.csv`). This recipe is inspired by Aerts et al. 2014; it is not their four-feature signature.

Median imputation and standardization are fitted on the training split only. An L2 Cox model is fitted on all covariates, without PCA, using Efron ties, L-BFGS and a Breslow baseline. The penalty is chosen from {0.01, 0.1, 1.0} by validation Uno C at 36 months, as in the frozen Linear Cox rung. The selected fit is trained on the training split only and is not refitted on train plus validation. The test fold is evaluated once.

The primary metric is pooled Uno C at 36 months over the 416 out-of-fold predictions, ranked by −log S(36 months). Paired intervals use 500 patient bootstrap resamples shared across all conditions, with the censoring curve fixed to the full cohort. The intervals are conditional on the fitted fold models and have no multiplicity adjustment. All E1/E2 numbers below were recomputed with this same estimator.

## Results

| Baseline | Pooled Uno C 36 m (95% CI) | Mean fold Uno C | AUC 1 / 2 / 3 y | IBS 3–36 m |
| --- | ---: | ---: | ---: | ---: |
| Stage + tumor volume | 0.557 (0.527–0.591) | 0.578 | 0.560 / 0.592 / 0.594 | 0.213 |
| GTV radiomics | 0.570 (0.538–0.602) | 0.577 | 0.614 / 0.596 / 0.601 | 0.210 |

Stage + volume varies widely across folds, from 0.435 in fold 4 to 0.637 in fold 3. Because separately fitted folds are ranked together, its pooled score is lower than its fold mean. Radiomics selected the strongest penalty, 1.0, in every fold. Radiomics minus stage + volume is +0.013 [−0.026, +0.051].

Paired pooled Uno C differences, FM condition minus baseline:

| Condition (pooled Uno C) | − Stage + volume | − Radiomics |
| --- | ---: | ---: |
| Linear COLIPRI (0.619) | **+0.062 [+0.014, +0.104]** | **+0.049 [+0.005, +0.087]** |
| Linear TANGERINE (0.595) | +0.038 [−0.009, +0.080] | +0.026 [−0.018, +0.070] |
| MLP COLIPRI (0.616) | **+0.059 [+0.013, +0.098]** | **+0.046 [+0.006, +0.082]** |
| MLP TANGERINE (0.611) | **+0.054 [+0.011, +0.093]** | +0.041 [−0.002, +0.082] |
| SQA TANGERINE (0.608) | **+0.051 [+0.009, +0.088]** | +0.038 [−0.000, +0.075] |
| SQA COLIPRI (0.605) | +0.048 [−0.001, +0.089] | +0.035 [−0.005, +0.076] |
| Linear Rad-JEPA-3D (0.528) | −0.028 [−0.071, +0.016] | −0.041 [−0.085, +0.001] |
| Linear CT-CLIP (0.513) | **−0.044 [−0.091, −0.001]** | **−0.057 [−0.104, −0.006]** |

All 40 contrasts, including LoRA and partial fine-tuning, are in `outputs/lung1_bf16_selected_baselines/paired_uno_c.json`. The LoRA and partial rungs score between 0.49 and 0.54 and all fall below the radiomics baseline.

## Interpretation and limits

On this split, frozen COLIPRI and TANGERINE heads score above both literature baselines. The intervals exclude zero for COLIPRI Linear and MLP against both baselines. Rad-JEPA-3D and CT-CLIP score at or below them. These results are exploratory, internal and unadjusted for multiplicity.

The baselines receive information the FMs do not: radiomics uses the GTV mask, and stage + volume uses clinical staging and the segmentation. LUNG1 was the radiomics discovery cohort in Aerts et al. and was already used in E1/E2 development. The baselines use ridge on all covariates, whereas the FM linear rung uses PCA32 with lifelines.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_bf16_selected_baselines.py
```

Outputs are under `outputs/lung1_bf16_selected_baselines/{stage_volume,radiomics}/`: `protocol.json`, `folds/`, `fold_metrics.csv`, `out_of_fold_predictions.csv` and `summary.json`. The paired comparison is in `paired_uno_c.json` and the log in `logs/lung1_bf16_selected_baselines/run.log`. Nothing was sent to W&B.
