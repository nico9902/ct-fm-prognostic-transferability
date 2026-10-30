# Phase 4: external validation on NSCLC-Radiogenomics

**Date:** 21 October 2026  
**Protocol:** [lung1_phase4_protocol.md](lung1_phase4_protocol.md). It was fixed before any image was processed, with three dated amendments, all made before any outcome was read.  
**Design:**
- 22 models refitted on the 416 LUNG1 patients and frozen, with hashes in `outputs/phase4_frozen_models/manifest.json`.
- Predictions computed from a copy of the cohort without outcome columns and hashed (`outputs/phase4_external/PREDICTED`).
- Applied **once**. Results are in `outputs/phase4_external/external_validation.json`.

## Summary

- **No primary contrast is significant**, as anticipated by the pre-registered power statement. The minimum detectable differences are 0.09–0.20.
- **Good generalist FMs transport; tumour-centred methods do not.**
  - TANGERINE, DINOv3 and COLIPRI are above chance in the larger secondary population (C 0.64–0.66, Holm p ≤ 0.003).
  - CT-CLIP and Rad-JEPA stay at chance, as in LUNG1.
  - Pai, Hosny zero-shot and Hosny frozen fall to chance (0.50–0.55).
- **Clinical covariates also transport, better than internally.** C1 goes from 0.565 to 0.646. The FMs add nothing measurable on top (C3-colipri − C1 = +0.007).
- **The ranking of the models transports.** The Spearman correlation between internal and external C across the 22 models is 0.67 (p = 0.0006).
- **Absolute calibration does not transport.** Observed/expected events at 36 months are about 0.4: the LUNG1 baseline hazard (stage III) strongly over-predicts mortality in early-stage surgical patients.

## Cohort

| | LUNG1 (training) | Radiogenomics primary | Radiogenomics secondary |
| --- | --- | --- | --- |
| Patients | 416 | 143 (with SEG) | 199 (all eligible CTs) |
| Deaths (by 36 months) | — | 50 (33) | 59 (41) |
| Median follow-up | — | 3.8 years | 3.3 years |
| Treatment / stage | radiotherapy, mostly stage III | surgery, 80% N0, mostly T1–T2 | |
| GTV volume, median | — | 8.1 cm³ | |
| Typical CT | 3 mm | 1–1.25 mm, often a sharp "lung" kernel | |

**Exclusions** (from the header audit, before any outcome was read):
- 9 patients had only PET/CT, attenuation-correction or whole-body CTs;
- 1 had a slice gap in the SEG-referenced CT (R01-119);
- 1 had a CT after the last contact.

All 143 tumour masks passed the checks. The labels are mislabelled by TCIA as "Heart"/"Tissue", but the contours were verified visually. The centroid is ≤ 0.52 mm from the Pai seed, and the mask overlaps the lung. C0 in the secondary population uses the 151 patients with pTNM (the AMC patients have none).

## Primary population (n = 143, 33 deaths by 36 months)

The metric is Uno C at 36 months, with 95% intervals from a 2,000-resample patient bootstrap. The internal LUNG1 value comes from phases 1–2 (ladder and E3 baselines on the locked folds). "Holm vs 0.5" is the chance family across the 22 models.

| Model | External C | 95% CI | Holm p vs 0.5 | Internal C | C 60 m | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- |
| C4-tangerine | 0.684 | [0.588, 0.776] | 0.002 | 0.604 | 0.666 | 1.10 |
| C3-dinov3 | 0.658 | [0.564, 0.756] | 0.026 | 0.578 | 0.642 | 1.74 |
| C3-colipri | 0.653 | [0.557, 0.748] | 0.027 | 0.599 | 0.621 | 1.84 |
| C4-colipri | 0.652 | [0.555, 0.751] | 0.040 | 0.606 | 0.652 | 0.92 |
| TANGERINE | 0.649 | [0.550, 0.741] | 0.041 | 0.599 | 0.606 | 0.62 |
| C1 (clinical + volume) | 0.646 | [0.547, 0.749] | 0.067 | 0.565 | 0.636 | 1.76 |
| DINOv3 | 0.643 | [0.556, 0.730] | 0.021 | 0.575 | 0.629 | 1.26 |
| C3-tangerine | 0.641 | [0.540, 0.738] | 0.072 | 0.598 | 0.603 | 0.55 |
| C2 (C1 + radiomics) | 0.633 | [0.539, 0.730] | 0.085 | 0.558 | 0.642 | 0.81 |
| C3-hosny_score | 0.627 | [0.527, 0.730] | 0.159 | 0.561 | 0.621 | 2.38 |
| COLIPRI | 0.627 | [0.540, 0.719] | 0.077 | 0.617 | 0.601 | 1.90 |
| C3-radjepa | 0.623 | [0.526, 0.720] | 0.159 | 0.562 | 0.613 | 1.40 |
| Radiomics baseline | 0.617 | [0.508, 0.723] | 0.299 | 0.591 | 0.625 | 0.45 |
| Stage + volume | 0.605 | [0.508, 0.702] | 0.367 | 0.588 | 0.609 | 3.24 |
| C3-ctclip | 0.602 | [0.497, 0.705] | 0.369 | 0.573 | 0.594 | 1.19 |
| C3-pai | 0.600 | [0.501, 0.697] | 0.367 | 0.557 | 0.575 | 1.00 |
| C0 (clinical) | 0.595 | [0.493, 0.698] | 0.402 | 0.529 | 0.598 | 1.73 |
| Pai | 0.547 | [0.446, 0.646] | 1.00 | 0.555 | 0.534 | 0.48 |
| Hosny zero-shot | 0.540 | [0.432, 0.643] | 1.00 | 0.566 | 0.541 | 0.13 |
| Rad-JEPA | 0.530 | [0.436, 0.627] | 1.00 | 0.539 | 0.540 | 0.05 |
| Hosny frozen | 0.498 | [0.400, 0.601] | 1.00 | 0.568 | 0.468 | 0.01 |
| CT-CLIP | 0.482 | [0.388, 0.581] | 1.00 | 0.530 | 0.470 | −0.57 |

Observed risk of death by 36 months: 25%. Predicted risk: about 60–85% depending on the model, so O/E is 0.29–0.67 (Hosny zero-shot has no survival curve).

### Pre-specified primary contrasts (Holm)

| Contrast | Difference [95% CI] | p | Holm p | MDD |
| --- | --- | --- | --- | --- |
| COLIPRI − stage + volume | +0.021 [−0.068, +0.126] | 0.65 | 1.00 | 0.134 |
| C3-colipri − C1 | +0.007 [−0.052, +0.071] | 0.83 | 1.00 | 0.088 |
| COLIPRI − Pai | +0.080 [−0.031, +0.195] | 0.17 | 0.67 | 0.162 |
| COLIPRI − Hosny zero-shot | +0.087 [−0.046, +0.231] | 0.22 | 0.67 | 0.198 |

No contrast is equivalent within ±0.03. That was not expected with 33 events.

Secondary contrasts against COLIPRI, descriptive only:

| Contrast | Difference [95% CI] |
| --- | --- |
| vs TANGERINE | −0.023 [−0.126, +0.083] |
| vs DINOv3 | −0.016 [−0.099, +0.074] |
| vs C1 | −0.020 [−0.111, +0.072] |
| vs CT-CLIP | +0.145 [+0.026, +0.272] |
| vs Rad-JEPA | +0.096 [−0.032, +0.228] |
| vs Hosny frozen | +0.129 [−0.010, +0.260] |

## Secondary population

| Population | Model | Uno C 36 m [95% CI] | Holm p vs 0.5 |
| --- | --- | --- | --- |
| All eligible CTs (n = 199, 41 deaths by 36 months) | TANGERINE | 0.662 [0.578, 0.745] | 0.0007 |
| | DINOv3 | 0.660 [0.580, 0.740] | 0.0004 |
| | COLIPRI | 0.641 [0.559, 0.726] | 0.003 |
| | CT-CLIP | 0.512 [0.419, 0.600] | 1.00 |
| | Rad-JEPA | 0.479 [0.398, 0.557] | 1.00 |
| With pTNM (n = 151, 36 deaths by 36 months) | COLIPRI | 0.632 [0.544, 0.717] | 0.006 |
| | C0 | 0.603 [0.508, 0.699] | 0.036 |
| | COLIPRI − C0 | +0.029 [−0.066, +0.126] | p = 0.56 |

## Interpretation

1. **The external result confirms the internal picture in its ordering, not in its magnitudes.**
   - The same three generalist FMs are useful, and the same two (CT-CLIP, Rad-JEPA) are not.
   - The tumour-centred methods are the ones that lose the most: Pai goes from 0.555 to 0.547, Hosny frozen from 0.568 to 0.498, and the Hosny score from 0.566 to 0.540. This agrees with phases 1–2: the primary tumour carries little prognostic signal, and what it carries is cohort-specific.
2. **The pre-registered hypothesis is supported in direction, not formally.** The FMs transport *at least as well as* stage + volume (0.627–0.649 vs 0.605), but the intervals overlap widely.
3. **Clinical data gain the most externally** (C0 +0.07, C1 +0.08). This is plausible: in early-stage surgical patients, age, T and N spread the risk more than in a homogeneous stage III cohort. Consequently, "FM over clinical" is still not demonstrated (+0.007).
4. **Which FM is "best" does not transport.** COLIPRI, the best FM internally, is overtaken externally by TANGERINE and DINOv3 (by about 0.02, well within the noise). The robust statement is about the *group* of good generalist FMs, not a single model.
5. **Calibration.** The calibration slopes vary widely (from 0.5 to 3), and O/E is about 0.4. Absolute risks from LUNG1 must not be used on a surgical early-stage population without recalibration.

## Limitations

- **Power.** There are 33 events by 36 months, and the minimum detectable differences between models are 0.09–0.20, so no equivalence or superiority claim between models is possible. The phase 4 numbers answer "above chance?" and "does the ordering hold?", not "is A better than B?".
- **The populations differ** (stage, treatment, pathological vs clinical TNM, CT kernel and slice thickness). This tests transportability under a strong shift, not a replication.
- **Each Uno C uses the Radiogenomics censoring distribution.** The bootstrap is conditional on the frozen models: training variability is not included.
- **Tumour-centred methods are evaluated only on the 143 patients with a SEG.**

## Files

- **Scripts:**
  - `scripts/build_radiogenomics_cohort.py`
  - `scripts/preprocess_radiogenomics.py`
  - `scripts/launch_radiogenomics_prep.sh`
  - `scripts/freeze_lung1_models.py`
  - `scripts/apply_radiogenomics_external.py`
- **Data:**
  - `data/derived/radiogenomics/{cohort.csv, cohort_with_gtv_volume.csv, ct_series_audit.csv}`
  - `data/processed/radiogenomics/thorax_hu/`
  - `outputs/radiogenomics_embeddings/`
- **Models and results:** `outputs/phase4_frozen_models/`, `outputs/phase4_external/`
