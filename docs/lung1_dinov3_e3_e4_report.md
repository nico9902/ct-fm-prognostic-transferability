# LUNG1 E3 and E4 regional probes: DINOv3 added

**Run date:** 17 October 2026  
**Cohort and split:** the same 416 patients (369 deaths), the five locked outer folds (repetition zero of `configs/lung1_e1_folds.csv`), the locked E3 subset file, and the same inner penalty selection as the [E3](lung1_e3_frozen_linear_report.md) and [E4 regional](lung1_e4_regional_frozen_cox_report.md) reports. Each outer development pool has 332–333 patients and each test fold has 83–84. Penalty selection uses 3 event-stratified inner folds within the development pool or budget. There is no fixed validation set. The [selected 5-fold E1/E2 split](lung1_dinov3_bf16_selected_report.md) (about 221/111/83 patients) is **not** used here, so DINOv3 compares directly with the other four encoders and the baselines.

DINOv3 ViT-B/16 (natural-image LVD-1689M pretraining) uses the same 2-D axial-slice preprocessing as in E1/E2. The global embedding is the slice mean of the final-norm CLS token (768 dimensions). Its saved tokens are the patch tokens on a slices × 14 × 14 grid, adaptively average-pooled to 8 × 8 × 8. No probe setting was changed for DINOv3. Adding DINOv3 to the E3 output directories left every earlier fit, subset and bootstrap result unchanged.

## E3: label-efficiency curves

Mean of five outer-fold Uno C through 36 months. Budgets below 100% are averaged over ten nested draws.

| Encoder | PCA | 10% (29 deaths) | 25% (74) | 50% (148) | 100% (295) | Log-event AULC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | 8 | 0.550 | 0.561 | 0.575 | 0.584 | 0.567 |
| TANGERINE | 8 | 0.539 | 0.547 | 0.562 | 0.572 | 0.554 |
| **DINOv3** | 8 | 0.517 | 0.524 | 0.535 | **0.566** | 0.532 |
| Rad-JEPA-3D | 8 | 0.511 | 0.516 | 0.524 | 0.536 | 0.520 |
| CT-CLIP | 8 | 0.522 | 0.520 | 0.529 | 0.533 | 0.525 |
| COLIPRI | 16 (post hoc) | 0.550 | 0.577 | 0.599 | 0.617 | 0.584 |
| TANGERINE | 16 | 0.558 | 0.576 | 0.595 | 0.599 | 0.582 |
| **DINOv3** | 16 | 0.528 | 0.534 | 0.555 | **0.575** | 0.545 |
| Rad-JEPA-3D | 16 | 0.515 | 0.525 | 0.526 | 0.539 | 0.525 |
| CT-CLIP | 16 | 0.512 | 0.521 | 0.527 | 0.530 | 0.522 |

The [summary figure](../outputs/lung1_e3_frozen_linear/e3_dinov3_summary.png) shows both PCA settings, with DINOv3 highlighted against the other encoders and the baselines (script `scripts/plot_lung1_e3_dinov3_summary.py`).

DINOv3's conditional 95% interval at 100% is 0.531–0.600 (PCA8) and 0.540–0.608 (PCA16). Its 10%-to-100% gain is +0.048 [+0.021, +0.076] with PCA8 and +0.047 [+0.023, +0.071] with PCA16. Across the ten draws, the SD of the 10% fold-mean score is 0.013 (PCA8) and 0.015 (PCA16). IBS 3–36 months improves from 0.235 to 0.212 (PCA8) and from 0.253 to 0.214 (PCA16).

DINOv3 sits near the two weaker 3-D encoders at small budgets and moves toward TANGERINE at 100% with PCA8. Its gain is the largest of the five with PCA8. Of the gain differences against DINOv3, only CT-CLIP minus DINOv3 has an interval that excludes zero (−0.038 [−0.077, −0.0005]). With PCA16, COLIPRI and TANGERINE are numerically above DINOv3 at every budget.

**Against the matched baselines** (same subsets; baseline minus DINOv3 PCA16, paired conditional 95% intervals):

| Baseline | 10% | 25% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: |
| Stage + tumor volume | +0.008 [−0.017, +0.033] | +0.028 [−0.007, +0.058] | +0.017 [−0.025, +0.052] | +0.013 [−0.034, +0.053] |
| GTV radiomics | +0.004 [−0.021, +0.028] | +0.031 [+0.002, +0.059] | +0.013 [−0.024, +0.044] | +0.016 [−0.028, +0.054] |

Both baselines are numerically above DINOv3 at every budget. Only the 25% radiomics interval excludes zero (with PCA8, also both baselines at 25%), with no multiplicity adjustment. The [updated matched plot](../outputs/lung1_e3_baselines/matched_learning_curves.png) now includes the DINOv3 PCA16 curve.

## E4: regional frozen-token probes

**Mapping.** The GTV and 0–20 mm shell were mapped onto DINOv3's native grid, with each token being one 3 mm slice × one 16 × 16 patch (median 3.0 × 35.7 × 35.7 mm in ZYX). The mapping was then adaptively average-pooled to 8³, the same way as the tokens. The geometry follows the official LPS orientation, 3 mm linear z resampling and 224 × 224 resizing. On test patients, a GTV pushed through the same torch interpolation correlates 0.9997–1.0 with the physical footprint, with matching totals. DINOv3 keeps the full in-plane field, so 100% of every GTV and shell lies inside its input field. The median saved-token count touched by the GTV is 8 (shell: 20); 14 patients have no shell-only block.

Mean held-out Uno C through 36 months, PCA16 (primary), and the same fits for the other four encoders:

| Model | All | Tumor | Non-GTV | Peri20 | Tumor + peri20 | Tumor − all (paired 95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | 0.591 | 0.597 | 0.589 | 0.604 | 0.605 | +0.006 [−0.019, +0.031] |
| TANGERINE | 0.582 | 0.573 | 0.574 | 0.593 | 0.590 | −0.010 [−0.054, +0.035] |
| Rad-JEPA-3D | 0.539 | 0.472 | 0.544 | 0.491 | 0.489 | −0.067 [−0.110, −0.025] |
| CT-CLIP | 0.526 | 0.561 | 0.518 | 0.547 | 0.553 | +0.035 [−0.015, +0.081] |
| **DINOv3** | 0.548 | **0.586** | 0.550 | 0.577 | 0.579 | **+0.038 [−0.005, +0.075]** |

For DINOv3, the tumor-weighted pool is its best region and also its best E3/E4 frozen representation: Harrell C 0.581 and IBS 0.209, against 0.549 and 0.215 for all tokens. The tumor-minus-all interval just includes zero. Tumor − non-GTV is +0.036 [−0.007, +0.074]. Adding the shell does not help (tumor + peri20 − tumor −0.007 [−0.020, +0.009]). Across encoders, the tumor-minus-all contrast of DINOv3 exceeds Rad-JEPA-3D's (difference of differences +0.105 [+0.057, +0.156]). It is not separable from COLIPRI (+0.032 [−0.020, +0.072]), TANGERINE (+0.048 [−0.001, +0.087]) or CT-CLIP (+0.003 [−0.054, +0.051]).

**Sensitivities.** With PCA8, tumor − all is +0.048 [−0.003, +0.095] and tumor − non-GTV is +0.054 [+0.003, +0.101]. Excluding LUNG1-038, -108 and -192 gives tumor − all +0.034 [−0.006, +0.078]. The tumor-position advantage of DINOv3 therefore has the same sign under all three settings but is not individually conclusive.

**Interpretation.** DINOv3 has no 3-D context and mixes information only within a slice, so its tumor-overlapping tokens come from fewer, tumor-containing slices. A slice-mean of all tokens dilutes them across about 120 slices. This is a likely reason why tumor-position pooling helps DINOv3 more than the 3-D encoders. As in the main E4 report, the regional pools use the GTV at inference and locate decodable information by token position, not by exclusive tissue source.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e3_frozen_linear.py --only-model dinov3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e3_frozen_linear.py --only-model dinov3 \
  --components 16 --output-dir outputs/lung1_e3_frozen_linear_pca16
.venv/bin/python scripts/analyze_lung1_e3_frozen_linear.py --bootstrap 300   # and --output-dir ..._pca16
.venv/bin/python scripts/analyze_lung1_e3_baselines.py --bootstrap 300
# Native-grid mapping (also writes outputs/lung1_e4_geometry_dinov3)
cd scripts && ../.venv/bin/python audit_lung1_e4_native_token_mapping.py --workers 8
../.venv/bin/python run_lung1_e4_regional_cox.py --models dinov3 \
  --geometry-dir ../outputs/lung1_e4_geometry_dinov3 --output-dir ../outputs/lung1_e4_regional_cox_dinov3
# also --components 8 (…_pca8_dinov3) and --exclude-patients LUNG1-038,LUNG1-108,LUNG1-192 (…_exclude3_dinov3)
../.venv/bin/python analyze_lung1_e4_regional_cox.py --output-dir ../outputs/lung1_e4_regional_cox_dinov3 \
  --merge-dir ../outputs/lung1_e4_regional_cox --write-dir ../outputs/lung1_e4_regional_cox_with_dinov3
```

Merged five-model regional results (paired differences, between-model interactions, plot) are in `outputs/lung1_e4_regional_cox{,_pca8,_exclude3}_with_dinov3`. The original four-model directories are unchanged. Baseline-versus-DINOv3 contrasts are in `outputs/lung1_e3_baselines/paired_dinov3_differences.csv`.
