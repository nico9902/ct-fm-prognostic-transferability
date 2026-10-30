# LUNG1: do frozen FM embeddings add to stage + tumor volume?

**Run date:** 17 October 2026  
**Cohort and split:** 416 patients (369 deaths). The split is the five locked outer folds of E3/E4 (repetition zero of `configs/lung1_e1_folds.csv`). Each model is fitted on the full development pool (332–333 patients) and evaluated on the untouched test fold (83–84). The penalty is chosen from 0.01, 0.1 and 1.0 by the same three event-stratified inner folds as the E3 full-budget fits.

## Design

| Model | Covariates |
| --- | --- |
| Clinical | Stage II, IIIa, IIIb and missing-stage indicators (stage I reference) plus log(1 + GTV volume in cm³), the [E3 stage + volume baseline](lung1_e3_matched_baselines_report.md) covariates |
| Clinical + FM | The same covariates plus the FM's frozen global E1 embedding, standardized and reduced to 16 PCA components (8 in the sensitivity analysis), all fitted on training patients only |
| FM only | The E3 full-budget linear Cox fit on the same folds and inner splits |

Clinical and clinical + FM use the same lifelines L2 Cox, with clinical covariates standardized. In the primary analysis a single selected penalty applies to all covariates, so a difference between the two comes only from adding the FM block. When the missing-stage indicator is constant in a training set (one patient has no stage), it is dropped from that fit. The refitted clinical model scores 0.587, matching the E3 baseline implementation (0.588).

**Primary test:** clinical + FM minus clinical. It uses the mean outer-fold Uno C through 36 months and PCA16 (the same primary choice as the [FM-versus-baseline tests](lung1_e3_matched_baselines_report.md#statistical-tests-every-fm-versus-the-baselines-17-october-2026)). The p-value is a two-sided Wald test with SE from 1,000 paired held-out patient resamples within each fold (seed 20261018), and Holm is applied over the five FMs. A secondary family asks the reverse question, clinical + FM minus FM only. The tests are conditional on the fitted models and the locked folds. This analysis was specified after the E3 and E4 results were known.

## Results (PCA16, primary)

Mean outer-fold Uno C at 36 months, 95% conditional bootstrap interval, full-follow-up Harrell C and IBS from 3 to 36 months:

| Model | Uno C (95% CI) | Harrell C | IBS |
| --- | ---: | ---: | ---: |
| Clinical | 0.587 (0.551–0.622) | 0.579 | 0.212 |
| Clinical + COLIPRI | 0.620 (0.583–0.654) | 0.615 | 0.206 |
| Clinical + TANGERINE | **0.621** (0.587–0.654) | 0.612 | **0.197** |
| Clinical + DINOv3 | 0.593 (0.557–0.629) | 0.589 | 0.211 |
| Clinical + Rad-JEPA-3D | 0.577 (0.540–0.613) | 0.571 | 0.210 |
| Clinical + CT-CLIP | 0.578 (0.540–0.614) | 0.572 | 0.211 |

**Added value of the FM** (clinical + FM minus clinical):

| FM | Δ Uno C [95% CI] | p | p<sub>Holm</sub> | Δ IBS |
| --- | --- | ---: | ---: | ---: |
| COLIPRI | +0.033 [−0.010, +0.069] | 0.095 | 0.379 | −0.007 |
| TANGERINE | +0.034 [−0.001, +0.071] | 0.067 | 0.333 | −0.015 |
| DINOv3 | +0.006 [−0.027, +0.041] | 0.726 | 1.000 | −0.001 |
| Rad-JEPA-3D | −0.010 [−0.038, +0.022] | 0.494 | 1.000 | −0.002 |
| CT-CLIP | −0.009 [−0.037, +0.024] | 0.557 | 1.000 | −0.001 |

**Added value of the clinical covariates** (clinical + FM minus FM only):

| FM | Δ Uno C [95% CI] | p<sub>Holm</sub> |
| --- | --- | ---: |
| COLIPRI | +0.003 [−0.011, +0.017] | 0.676 |
| TANGERINE | +0.022 [+0.005, +0.041] | **0.040** |
| DINOv3 | +0.018 [−0.005, +0.042] | 0.236 |
| Rad-JEPA-3D | +0.038 [+0.012, +0.065] | **0.025** |
| CT-CLIP | +0.048 [+0.018, +0.078] | **0.010** |

The Spearman correlation between held-out FM-only risk and clinical risk is 0.28 for COLIPRI, 0.16 for TANGERINE, 0.14 for DINOv3, 0.11 for Rad-JEPA-3D and 0.04 for CT-CLIP. COLIPRI's combined model selected the largest penalty (1.0) in all five folds.

## Sensitivity analyses

- **PCA8.** All gains shrink to between −0.008 and +0.006 (all p<sub>Holm</sub> = 1.000). COLIPRI and TANGERINE reach 0.593.
- **Clinical covariates lightly penalized.** Here the clinical block gets a fixed penalty of 0.01 and the grid tunes only the FM block, so the clinical coefficients are not shrunk with the FM. The clinical reference drops to 0.580. The gains are +0.025 for COLIPRI (p<sub>Holm</sub> 0.586) and +0.032 [+0.001, +0.066] for TANGERINE (p 0.052, p<sub>Holm</sub> 0.261); the other three stay within ±0.007.

## Interpretation

- **No FM adds significant discrimination to stage + volume after multiplicity correction.** COLIPRI and TANGERINE give the largest and most consistent gains, about +0.03 Uno C at PCA16. TANGERINE also gives the largest IBS improvement. These gains are compatible with zero, smaller at PCA8, and need a larger or external cohort to confirm. Rad-JEPA-3D, CT-CLIP and DINOv3 add nothing.
- **The reverse direction separates the encoders.** Stage + volume adds little to COLIPRI (+0.003), but significantly improves TANGERINE, Rad-JEPA-3D and CT-CLIP. COLIPRI's embedding therefore already carries most of the clinically captured risk; its risk score is also the most correlated with the clinical score. TANGERINE's signal is more complementary: neither source alone reaches the combined score.
- Together with the E4 interventions, this suggests that the frozen FMs mainly encode disease-extent information related to stage and tumor burden. Only the two strongest encoders show any evidence of further prognostic information, which is not established here.
- **Limits.** This is internal cross-validation on LUNG1, with no external validation. The analysis was specified after E3 and E4. With one shared penalty and three grid values, COLIPRI selected the edge of the grid. The clinical model has only five covariates, and stage is missing for one patient.

## Reproduction

```bash
cd scripts
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python run_lung1_clinical_plus_fm_cox.py --components 16
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python run_lung1_clinical_plus_fm_cox.py --components 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python run_lung1_clinical_plus_fm_cox.py --components 16 --clinical-penalty 0.01
```

Fits (with coefficients and held-out predictions), `summary.csv` and `tests.csv` are in `outputs/lung1_clinical_plus_fm/{pca16,pca8,pca16_clinpen0.01}`.
