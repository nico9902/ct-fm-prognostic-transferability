# LUNG1 phase 1: pre-specified protocol

**Fixed on:** 19 October 2026, before any phase 1 result was computed  
**Scope:** empirical ceiling ladder (1.1), power (1.2), embedding probing (1.3), E4 without refitting (1.4), residual FM signal (1.5)

## Common setup

- **Cohort:** the 416 LUNG1 patients of E1–E4 (`data/derived/lung1/cohort_with_gtv_volume.csv`).
- **Split:** repetition 0 of the locked folds (`configs/lung1_e1_folds.csv`), five outer folds. Within each outer development set, the same three event-stratified inner folds as E3/E4 select the penalties. This is the same split as E4 and the clinical + FM report, so their numbers are directly comparable.
- **Primary metric:** mean outer-fold Uno C through 36 months.
- **Intervals:** paired patient bootstrap within each fixed fold (1,000 resamples, seed 20261019). They are conditional on the fitted models and folds.

## 1.1 Ceiling ladder

| Model | Covariates |
| --- | --- |
| C0 | Age (median-imputed, with a missing indicator), male sex, T2/T3/T4, N1/N2/N3, stage II/IIIa/IIIb/missing, histology (large cell, adenocarcinoma, NOS, missing; squamous is the reference) |
| C1 | C0 + log(1 + GTV volume in cm³) |
| C2 | C1 + GTV radiomics (107 PyRadiomics Original features, PCA16) |
| C3-*fm* | C1 + frozen global embedding (PCA16), for each of the five FMs |
| C4-colipri | C1 + radiomics + COLIPRI; **pre-specified primary C4** |
| C4-tangerine | Sensitivity |

**Coding.** T5 (2 patients) is merged into T4, and N4 (3 patients) into N3. M is excluded, because 411 of 416 patients are M0. The covariates were chosen so that they can be mapped onto NSCLC-Radiogenomics for the external validation.

**Fitting.** Each model is an L2 Cox (lifelines). The clinical block has its own penalty. The radiomics and image blocks share a second penalty. Both penalties are chosen from {0.01, 0.1, 1} (a 3 × 3 grid) by mean inner Uno C. This replaces the "light fixed clinical penalty" in the TODO, so that no clinical penalty is chosen by hand. All imputation, scaling and PCA are fitted on training patients only.

**Sensitivity.** Radiomics with all 107 features instead of PCA16.

**Contrast families.** Holm correction is applied within each family:

| Family | Contrasts |
| --- | --- |
| Primary | C1 − C0, C2 − C1, C3-colipri − C1, C4-colipri − C3-colipri, C4-colipri − C2 |
| Image over C1 (1.5) | C3-*fm* − C1, for each of the five FMs |
| C1 over image (1.5) | C3-*fm* − *fm* alone (the E3 PCA16 full-budget fits on the same folds) |
| Radiomics over C3 | C4-*fm* − C3-*fm* |
| Image versus radiomics | C3-*fm* − C2 |

**Tests.**
- **Difference:** two-sided Wald test with the bootstrap SE.
- **Equivalence:** TOST with a margin of **±0.03 Uno C**. Equivalence is declared when the 90% percentile interval lies inside the margin.
- **Empirical ceiling:** the maximum Uno C over the ladder.

## 1.2 Power

For each contrast, the minimum detectable difference is (z₀.₉₇₅ + z₀.₈) × the paired bootstrap SE, which corresponds to 80% power at a two-sided 5% level.

## 1.3 Embedding probing

- **Inputs:** each FM's official global vector from the E4 token-masking extraction, in two conditions: original, and GTV tokens hidden.
- **Targets:**
  - continuous: age, log GTV volume, scored by Spearman ρ (R² is also reported);
  - binary: male, T3–T4, N+, N2–N3, stage III, squamous, adenocarcinoma (both histology targets exclude NOS and missing), scored by ROC AUC.
- **Probes:** standardization, then either PCA16 (primary, the same capacity as the Cox probes) or all dimensions, followed by ridge or L2 logistic regression with 3-fold inner CV.
- **Uncertainty:** 1,000-resample patient bootstrap. The original − masked difference is paired.

## 1.4 E4 without refitting

- **Fit:** the PCA16 Cox from E4, fitted on original vectors with the penalty E4 selected.
- **Check:** the refit must reproduce the saved E4 risks exactly.
- **Application:** the model is then applied unchanged to the GTV-masked vectors and, on the 403 patients with a matched control, to the control-masked vectors.
- **Readouts:** the retained-token mean and the global vector.
- **Contrasts:** original − masked without refitting, and control-masked − GTV-masked without refitting.

## Execution

```bash
bash scripts/launch_lung1_phase1.sh   # detached; logs in outputs/lung1_phase1_logs/
```
