# LUNG1 E3: matched clinical and radiomics baselines

**Run date:** 9 October 2026  
**Cohort and splits:** the same 416 patients, five locked outer folds, ten nested draws at 10%, 25%, and 50%, and one full-development fit per fold as [E3](lung1_e3_frozen_linear_report.md). All 310 baseline fits completed.

## Baselines and comparison protocol

- **Stage + tumor volume:** stage II, IIIa, and IIIb indicators (stage I reference), one missing-stage indicator, and log(1 + GTV volume in cm³). One of the 416 eligible patients has no recorded overall stage. No outcome is used to construct these covariates.
- **GTV radiomics:** 107 finite, nonconstant PyRadiomics 3.0.1 `Original` image features from the cached 1 mm isotropic HU CT and primary-neoplasm mask. CT is cropped around the GTV with a five-voxel margin before feature extraction; this matched the full-image features to numerical precision on a checked patient. Intensity discretization uses a 25 HU bin width. No wavelet or LoG filter features are included. This is a practical radiomics baseline inspired by [Aerts et al.'s LUNG1 radiomics study](https://www.nature.com/articles/ncomms5006), not a reproduction of its selected signature or its published score.

**Published reference:** Aerts et al. trained a four-feature Cox signature on Lung1 (422 patients), then reported a **C-index of 0.65 on the independent Lung2 cohort** (225 patients). One signature feature was wavelet based. The 0.65 is an external Lung2 result, whereas every score below is from internal held-out LUNG1 folds. Our 107-feature PyRadiomics recipe and fitted Cox model also differ from their selected four-feature signature. Their 0.65 therefore gives useful context but is not a point on the matched E3 curve. [Original paper and Figure 4](https://www.nature.com/articles/ncomms5006).

Both baselines use training-only standardization and L2-regularized linear Cox regression on all their covariates, without PCA. They use the E1 ridge Cox L-BFGS implementation with Efron ties; the primary E3 FM probes use lifelines Cox fitting after PCA. A penalty from 0.01, 0.1, and 1.0 is selected by three event-stratified inner folds **within each E3 label budget**, maximizing Uno C through 36 months. If a feature is missing, its median is estimated using only the fitting patients in each inner or final fit. The final model is refitted on the full sampled subset, then evaluated on the same untouched E3 outer test patients. For held-out Uno C, AUC, and IBS, the censoring curve comes from the corresponding full outer development pool solely to keep evaluation weights fixed across budgets. It does not enter fitting or penalty selection.

The radiomics model receives a tumor segmentation at inference; the E3 frozen FM embeddings came from each encoder's original CT preprocessing without the GTV mask as input. This difference matters when interpreting relative performance. The graph below displays the **16-component FM sensitivity** requested for comparison with the baselines. The original E3 primary analysis used eight PCA components. The 16-component run was examined after seeing the primary results, so conclusions based on it remain exploratory. The baselines use their full covariates without PCA.

## Held-out results

The primary metric is the mean of five outer-fold Uno C-indices through 36 months, averaged over ten subset draws at budgets below 100%. Counts below are mean observed deaths over full follow-up in the fitting subset, as in E3.

| Model | 10% (29 deaths) | 25% (74 deaths) | 50% (148 deaths) | 100% (295 deaths) |
| --- | ---: | ---: | ---: | ---: |
| Stage + tumor volume Cox | 0.537 | 0.562 | 0.572 | 0.588 |
| GTV radiomics Cox | 0.533 | 0.565 | 0.568 | 0.591 |
| COLIPRI, post hoc PCA16 | 0.550 | **0.577** | **0.599** | **0.617** |
| TANGERINE, post hoc PCA16 | **0.558** | 0.576 | 0.595 | 0.599 |
| Rad-JEPA-3D, post hoc PCA16 | 0.515 | 0.525 | 0.526 | 0.539 |
| CT-CLIP, post hoc PCA16 | 0.512 | 0.521 | 0.527 | 0.530 |

The [matched learning-curve plot](../outputs/lung1_e3_baselines/matched_learning_curves.png) shows the two baselines and all four **PCA16** FM curves. Baseline shading gives 95% patient-bootstrap intervals conditional on the fitted models, fixed folds, and fixed subset draws. The separate [E3 report](lung1_e3_frozen_linear_report.md) gives FM uncertainty bands and the original PCA8 analysis.

At 100% training, stage + volume trails PCA16 COLIPRI by 0.029 Uno C (paired conditional 95% interval −0.071 to 0.015); radiomics trails it by 0.026 (−0.072 to 0.015). At 10%, the differences are −0.013 (−0.040 to 0.013) and −0.017 (−0.041 to 0.005). Every baseline-versus-PCA16-COLIPRI interval in [the complete difference table](../outputs/lung1_e3_baselines/paired_colipri_differences.csv) includes zero. The original PCA8 COLIPRI curve and its paired differences are also retained in the E3 artifacts; at 100% it scored 0.584. Thus the apparent gap to the baselines depends on probe capacity, and the PCA16 comparison is exploratory.

Radiomics exceeds stage + volume by only 0.003 Uno C at 100% (paired conditional 95% interval −0.030 to 0.036). Their [paired differences at all budgets](../outputs/lung1_e3_baselines/paired_baseline_differences.csv) also include zero.

## Secondary metrics and limits

| Baseline | Full-follow-up Harrell C, 10% → 100% | Full-follow-up Uno C, 10% → 100% | IBS 3–36 months, 10% → 100% |
| --- | ---: | ---: | ---: |
| Stage + tumor volume | 0.533 → 0.579 | 0.532 → 0.577 | 0.223 → 0.212 |
| GTV radiomics | 0.528 → 0.578 | 0.527 → 0.575 | 0.299 → 0.208 |

Radiomics has the lowest full-budget IBS of these two baselines, but its 10% IBS is much worse (0.299). Its small-budget survival probabilities should not be interpreted as well calibrated from concordance alone. The optional per-fold calibration-slope regression issued one convergence warning; all 310 primary Uno C results are finite, and that diagnostic regression is separate from the penalized Cox fits.

This is internal cross-validation on a cohort already used in earlier E1/E2 development. The bootstrap intervals resample held-out patients within locked folds and do not include model refitting, alternative subset draws, or external-cohort uncertainty. Aerts et al. used LUNG1 as their radiomics discovery cohort, so their reported performance is not an independent test-set benchmark for this experiment.

## Reproduction and artifacts

The main project environment remains on NumPy 2.2.6. PyRadiomics 3.0.1 requires NumPy 1.x in this setup, so feature extraction uses an isolated CPU environment:

```bash
python3 -m venv outputs/lung1_e3_baselines/venv_radiomics
outputs/lung1_e3_baselines/venv_radiomics/bin/python -m pip install -r requirements-radiomics.txt
outputs/lung1_e3_baselines/venv_radiomics/bin/python -m pip install --no-build-isolation pyradiomics==3.0.1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  outputs/lung1_e3_baselines/venv_radiomics/bin/python scripts/extract_lung1_e3_radiomics.py --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e3_baselines.py
MPLCONFIGDIR=/tmp/lung1_e3_baselines_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_lung1_e3_baselines.py --bootstrap 300
```

The [baseline output directory](../outputs/lung1_e3_baselines) contains the feature table and provenance, 310 per-fit JSON records, held-out predictions, fold/draw/summary tables, paired intervals, plot, and completion marker. Patient-level outputs remain local under `outputs/`.

## Statistical tests: every FM versus the baselines (17 October 2026)

All five frozen FMs, now including DINOv3, are tested against both baselines at every budget, with PCA8 and PCA16. The tests reuse the 300 paired resamples and seed of the intervals above. p-values are two-sided Wald tests with z = difference / bootstrap SE, the same method as the E1/E2 tests. Holm is applied within each family of 10 contrasts (5 FMs × 2 baselines) at one budget and PCA setting. **The primary family is 100% with PCA16.** It was declared after the E3 results were known, and PCA16 was itself a post hoc sensitivity. All other families are exploratory. Tests are conditional on the fitted models, folds and subset draws.

Primary family, FM minus baseline at 100% budget with PCA16 [95% CI]; p (unadjusted); p<sub>Holm</sub>:

| FM | − Stage + volume | − Radiomics |
| --- | --- | --- |
| COLIPRI | +0.029 [−0.015, +0.071]; 0.198; 1.000 | +0.026 [−0.015, +0.072]; 0.233; 1.000 |
| TANGERINE | +0.011 [−0.031, +0.057]; 0.630; 1.000 | +0.008 [−0.030, +0.052]; 0.708; 1.000 |
| DINOv3 | −0.013 [−0.053, +0.034]; 0.577; 1.000 | −0.016 [−0.054, +0.028]; 0.474; 1.000 |
| Rad-JEPA-3D | −0.049 [−0.092, +0.004]; 0.045; 0.315 | −0.052 [−0.096, −0.000]; 0.034; 0.275 |
| CT-CLIP | −0.058 [−0.109, −0.007]; 0.025; 0.221 | −0.061 [−0.114, −0.009]; 0.015; 0.145 |

No FM differs significantly from either baseline in the primary family. Across all 80 contrasts, no FM is significantly *above* a baseline in any family. The only Holm-significant contrasts are PCA8 at 25%: Rad-JEPA-3D − stage + volume (p<sub>Holm</sub> 0.048), Rad-JEPA-3D − radiomics (0.009) and CT-CLIP − radiomics (0.037), all below the baselines. Every contrast is in `outputs/lung1_e3_baselines/statistical_tests_fm_vs_baselines.{csv,json}` (`scripts/test_lung1_e3_baselines_stats.py`).
