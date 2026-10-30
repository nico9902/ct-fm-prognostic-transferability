# LUNG1 phase 2: tumour-centred methods (Pai 2024, Hosny 2018)

**Date:** 21 October 2026  
**Protocol:** [lung1_phase2_protocol.md](lung1_phase2_protocol.md), fixed before the results, including the recorded deviation on the Hosny reproduction check.  
**Cohort and split:** the 416 LUNG1 patients, repetition 0 of the locked folds. The split is the same as for the five whole-CT FMs.

## Summary

- **Neither tumour-centred method beats COLIPRI.** On every selected-protocol rung (PCA32, MLP), both Pai and Hosny are 0.05–0.09 Uno C below COLIPRI. All six contrasts remain significant after Holm. **Caveat:** in E3 with PCA8, Pai equals COLIPRI (0.583 vs 0.584). The gap for Pai therefore depends on the probe capacity (see E3).
- **They add nothing to clinical data and volume.** Pai and Hosny are indistinguishable from stage + volume and from GTV radiomics. Added to C1, they do not raise C (C3-pai − C1 = −0.007, C3-hosny − C1 = −0.011).
- **Fine-tuning Hosny on LUNG1 makes it worse.** The best version of Hosny is the zero-shot model trained on HarvardRT: Uno C 0.569. Fine-tuning gives 0.531 (fine-tuned − zero-shot = −0.037, p = 0.034, Holm 0.14). The epoch selected on validation does not transfer to the test fold: in fold 4, validation is 0.640 and test is 0.482.
- **The phase 1 reading is confirmed.** The prognostic signal that the frozen FMs capture is not in the primary tumour. A dedicated model on the tumour patch, whether a tumour-specific foundation model (Pai) or a supervised prognostic CNN (Hosny), recovers about what the radiomics or the volume recover (~0.55–0.57), not what the whole-CT embedding recovers (~0.60–0.62).

## Extraction (2.2)

- **Seed:** the GTV centroid. Distance from the Pai repository LUNG1 annotations: median 0.16 mm, 95th percentile 0.36 mm, max 3.6 mm.
- **Outputs:** Pai gives 4096-d vectors and Hosny 256-d vectors (dense_2), plus the zero-shot log-odds. They are in `outputs/lung1_e1_embeddings/{pai,hosny,hosny_score}/`.
- **The Hosny reproduction check failed.** Spearman with the published Maastro predictions is 0.765 (n = 205), below the 0.9 threshold. The investigation excludes ids, orientation, seed and interpolation, and attributes the gap to the authors' input arrays. Our 2-year AUC on the same 205 patients is 0.667, vs 0.695 published (difference −0.028 [−0.084, +0.025]). Hosny is therefore reported as a *reimplementation on the public TCIA data*. Details are in the protocol.

## E1/E2 on the selected protocol (2.3)

The metric is pooled out-of-fold Uno C at 36 months, with a 500-resample paired bootstrap and the same estimator and seed as `statistical_tests.json`. Results are in `outputs/lung1_bf16_selected/statistical_tests_tumour_centred.json`.

| Condition | Uno C 36 m [95% CI] | vs COLIPRI on the same rung | Holm p |
| --- | --- | --- | --- |
| Linear Cox · COLIPRI (reference) | 0.619 [0.582, 0.653] | — | — |
| MLP Cox · COLIPRI (reference) | 0.616 [0.584, 0.647] | — | — |
| Linear Cox · Pai | 0.543 [0.505, 0.574] | −0.076 | 0.006 |
| MLP Cox · Pai | 0.552 [0.518, 0.584] | −0.063 | 0.007 |
| Linear Cox · Hosny | 0.537 [0.502, 0.579] | −0.082 | 0.004 |
| MLP Cox · Hosny | 0.540 [0.503, 0.578] | −0.076 | 0.004 |
| Hosny fine-tuned | 0.531 [0.498, 0.562] | −0.088 (vs linear) | 0.001 |
| **Hosny zero-shot** | **0.569 [0.534, 0.602]** | −0.050 (vs linear) | 0.038 |
| Baseline · stage + volume | 0.557 [0.528, 0.591] | | |
| Baseline · GTV radiomics | 0.570 [0.538, 0.602] | | |

- **Against the baselines:** no Pai or Hosny condition differs from stage + volume or from radiomics. The differences range from −0.038 to +0.012, and every Holm p is ≥ 0.67.
- **Against chance (Holm):** the zero-shot model (p = 0.0005) and MLP · Pai (p = 0.010) are above 0.5. Linear · Pai (p = 0.055), linear and MLP · Hosny, and fine-tuned Hosny (p ≈ 0.10–0.11) are not.
- **Hosny fine-tuning** (FP32, V100):
  - Selected epochs per fold: 68, 4, 24, 70, 10.
  - Test fold Uno C: 0.520, 0.554, 0.531, 0.630, 0.482.
  - Fine-tuned − zero-shot: −0.037 (p = 0.034, Holm 0.14).
  - Fine-tuned − linear on its frozen embedding: −0.005.
  - **Reading:** with about 250 training patients, fine-tuning the CNN is not stable. Epoch selection on about 80 validation patients picks noise. This mirrors the LoRA/partial collapse of the whole-CT FMs.
- **MLP vs linear:** no gain for either method (Pai +0.009, Hosny +0.003).

## E3 label efficiency (2.4)

Frozen linear Cox, PCA16, mean outer-fold Uno C at 36 months (`outputs/lung1_e3_frozen_linear_pca16`):

| Model | 10% | 25% | 50% | 100% |
| --- | --- | --- | --- | --- |
| COLIPRI | 0.550 | 0.577 | 0.599 | 0.617 |
| TANGERINE | 0.558 | 0.576 | 0.595 | 0.599 |
| DINOv3 | 0.528 | 0.534 | 0.555 | 0.575 |
| **Hosny** (frozen dense_2) | 0.512 | 0.539 | 0.555 | 0.568 |
| **Pai** | 0.531 | 0.542 | 0.545 | 0.555 |
| Rad-JEPA | 0.515 | 0.525 | 0.526 | 0.539 |
| CT-CLIP | 0.512 | 0.521 | 0.527 | 0.530 |

- **Pai** is relatively good with few labels (0.531 at 10%, above DINOv3), but its curve is almost flat from 25% onwards. The tumour-patch information saturates quickly.
- **Hosny frozen** keeps rising up to 100%, but stays below the two best FMs at every budget.
- Fine-tuned Hosny was not run in E3, a decision written in the protocol. E4 does not apply.
- **Sensitivity to the PCA dimension.** The PCA8 run is the original E3 table (`outputs/lung1_e3_frozen_linear`):

| Model (PCA8) | 10% | 25% | 50% | 100% |
| --- | --- | --- | --- | --- |
| COLIPRI | 0.550 | 0.561 | 0.575 | 0.584 |
| **Pai** | 0.544 | 0.555 | 0.566 | **0.583** |
| TANGERINE | 0.539 | 0.547 | 0.562 | 0.572 |
| DINOv3 | 0.517 | 0.524 | 0.535 | 0.566 |
| **Hosny** | 0.514 | 0.527 | 0.539 | 0.549 |
| Rad-JEPA | 0.511 | 0.516 | 0.524 | 0.536 |
| CT-CLIP | 0.522 | 0.520 | 0.529 | 0.533 |

  **With PCA8, Pai reaches COLIPRI** (0.583 vs 0.584) and sits above TANGERINE at every budget. The gap at PCA16 comes mostly from COLIPRI and TANGERINE improving with more components (+0.033 and +0.027), while Pai gets worse (−0.028). Pai's prognostic information is concentrated in a few directions; the whole-CT FMs spread theirs over more.
  - The primary comparison stays PCA16, as fixed in phase 0. On the selected protocol (PCA32) and with the MLP, Pai is also clearly below COLIPRI.
  - For Pai, though, the gap from the best FMs depends on the capacity of the probe. The paper must say "at matched capacity, Pai is at most on par with COLIPRI, never better", not "Pai is clearly worse".

## Ceiling ladder (2.5)

The output is in `outputs/lung1_ceiling_ladder/primary_tumour_centred`. The refit of every phase 1 model reproduces `summary.csv` and `tests.csv` exactly.

| Model | Uno C 36 m [95% CI] |
| --- | --- |
| C1 (clinical + volume) | 0.565 [0.531, 0.598] |
| C2 (C1 + radiomics) | 0.558 [0.523, 0.594] |
| C3-colipri | 0.599 [0.563, 0.635] |
| C3-pai | 0.557 [0.521, 0.591] |
| C3-hosny | 0.554 [0.518, 0.588] |
| C3-hosny_score | 0.561 [0.526, 0.595] |
| Pai alone (E3 PCA16) | 0.555 [0.518, 0.591] |
| Hosny alone (E3 PCA16) | 0.569 [0.532, 0.602] |
| Hosny zero-shot (no fit) | 0.566 [0.529, 0.600] |
| COLIPRI alone | 0.617 [0.582, 0.652] |

Contrasts, as difference [95% CI], with Holm p within each family:

| Family | Contrast | Difference | Holm p | Equivalent within ±0.03 |
| --- | --- | --- | --- | --- |
| tumour_centred_over_C1 | C3-pai − C1 | −0.007 [−0.036, +0.020] | 1.00 | no |
| | C3-hosny − C1 | −0.011 [−0.036, +0.014] | 1.00 | no |
| | C3-hosny_score − C1 | −0.004 [−0.020, +0.013] | 1.00 | **yes** |
| tumour_centred_vs_radiomics | C3-pai − C2 | −0.001 [−0.031, +0.031] | 1.00 | **yes** |
| | C3-hosny − C2 | −0.005 [−0.039, +0.027] | 1.00 | no |
| | C3-hosny_score − C2 | +0.002 [−0.027, +0.033] | 1.00 | **yes** |
| tumour_vs_whole_ct | C3-pai − C3-colipri | −0.042 [−0.079, −0.005] | 0.14 | no |
| | C3-hosny − C3-colipri | −0.046 [−0.085, −0.007] | 0.14 | no |
| | C3-hosny_score − C3-colipri | −0.039 [−0.076, −0.000] | 0.14 | no |
| | pai − colipri | −0.062 [−0.107, −0.015] | 0.056 | no |
| | hosny − colipri | −0.048 [−0.099, −0.003] | 0.14 | no |
| | hosny_zero_shot − colipri | −0.051 [−0.097, −0.005] | 0.14 | no |

How to read these contrasts:
- **The Hosny score adds nothing to C1:** the two are equivalent within ±0.03.
- **Pai and the Hosny score carry the same information as the radiomics:** equivalent within ±0.03.
- **Against COLIPRI:** all six "tumour vs whole CT" contrasts are negative (−0.04 to −0.06), with no 95% interval crossing zero. Holm across the six contrasts leaves them at p 0.06–0.14. On the selected protocol, with pooled estimation, the same comparison is significant after Holm (table in 2.3).

## Implications for the paper

1. **The tumour-centred group is not competitive.** "Tailored" approaches that use the tumour location (radiomics, Pai, Hosny) all sit at about 0.55–0.57. The best FMs on the whole CT, without that information, sit at about 0.60–0.62. The asymmetry favours the tumour-centred methods, which makes the result stronger.
2. **This agrees with phase 1.** The probing (1.3) and E4 without refit (1.4) showed that the prognostic signal of the FMs passes through age, sex, anatomy and nodal involvement, not through the primary GTV.
3. **Hosny:**
   - The zero-shot score (0.569 here; AUC 0.60 at 2 years on the full cohort) is below the paper's AUC of 0.70. The paper's value refers to the authors' 205-patient subset and their input arrays.
   - Fine-tuning on LUNG1 does not help.
   - It is reported as a reimplementation, with the failed check.
4. **External validation (phase 4):** Pai and Hosny need the segmentation, available for 144 Radiogenomics patients. Fine-tuned Hosny is not carried into phase 4 as a primary model, because it is worse than the zero-shot model on LUNG1.

## Files

- **Scripts:**
  - `scripts/extract_lung1_tumor_patch_embeddings.py`
  - `scripts/run_lung1_hosny_finetune.py`
  - `scripts/run_lung1_bf16_selected_heads.py` (`--model pai|hosny`)
  - `scripts/run_lung1_e3_frozen_linear.py`
  - `scripts/run_lung1_ceiling_ladder.py --tumor-centred`
  - `scripts/test_lung1_phase2_selected_stats.py`
  - `scripts/launch_lung1_phase2.sh`
- **Logs:** `outputs/lung1_phase2_logs/`, `outputs/logs/lung1_phase2_*.log`
