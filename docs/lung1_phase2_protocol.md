# LUNG1 phase 2: pre-specified protocol (tumour-centred methods)

**Fixed on:** 21 October 2026, after the extraction started and before any phase 2 result was computed  
**Scope:** Pai et al. 2024 (FMCIB, tumour-centred foundation model) and Hosny et al. 2018 (tailored deep CNN) on the same LUNG1 split as the five whole-CT FMs (TODO 2.2–2.5).

## 2.2 Extraction

- **Seed:** physical centroid of the cached primary GTV mask. This is the same mask used for the volume, the radiomics and E4.
- **Pai:** official FMCIB preprocessing (50³ at 1 mm, LPS, (HU + 1024) / 3072), MONAI ResNet50 trunk, 4096-d vector.
- **Hosny:** 50³ central crop of a 150³, 1 mm grid centred on the seed, HU [-1024, 3071] → [0, 1]. The PyTorch port gives a 256-d dense_2 vector and the zero-shot log-odds of death at 2 years.
- **Pre-specified check:** our preprocessing is kept if the Spearman correlation between our zero-shot P(dead at 2 years) and the authors' published Maastro (LUNG1) predictions is ≥ 0.9. If it is below 0.9, we stop and investigate before using the zero-shot score. The seed distance to the Pai repository LUNG1 annotations is reported.

## 2.3 E1/E2 frozen heads and Hosny fine-tuning

These use the same selected protocol as the whole-CT FMs (`outputs/lung1_bf16_selected`): repetition 0, five outer folds, and inner fold 0 of each development set as validation. All settings are the ones the other encoders used:

- **Linear Cox:** PCA32, L2 penalty from {0.01, 0.1, 1} on validation.
- **MLP Cox:** hidden 32, dropout 0.1, lr 3e-4, wd 1e-4, 100 epochs. The epoch is chosen by maximum validation Uno C at 36 months, with the earliest epoch winning ties. Lightning BF16 on the A100.
- **SQA:** not applicable, because there are no spatial tokens.
- **Hosny fine-tuned** (the only fine-tuned model, as the tailored method):
  - All weights start from HarvardRT, so epoch 0 is the zero-shot model and is a candidate for selection.
  - Exact full-batch Efron loss, AdamW, **lr 1e-4 fixed**, wd 1e-4, 100 epochs.
  - Eval mode: no dropout, frozen BatchNorm statistics.
  - Same selection rule as the MLP. FP32 on a V100.
  - *Decision:* there is no learning-rate sweep. The LoRA/partial rungs of the FMs also used one fixed setting each, so a sweep only for Hosny would favour the tailored method. It would also turn the selection into a hidden tuning step on the validation fold.

## 2.4 E3 label efficiency

- Pai and Hosny (frozen embeddings) are appended to the existing E3 runs, with identical subsets, draws, budgets (10/25/50/100%) and penalty selection.
- **PCA16** (`outputs/lung1_e3_frozen_linear_pca16`) is the primary run. **PCA8** (`outputs/lung1_e3_frozen_linear`) is the original E3 table. The protocol check refuses the append if anything already locked differs.
- *Decision:* **fine-tuned Hosny is not run in E3.** E3 is defined on frozen embeddings with linear Cox. Fine-tuning at every budget and draw (5 folds × 31 fits) is a different experiment, and it is the one method that would get trained weights. The frozen Hosny embedding and the zero-shot score represent Hosny in E3.
- E4 does not apply: the input is already tumour-centred.

## 2.5 Ceiling ladder

- New models:
  - C3-pai and C3-hosny: C1 + embedding, PCA16.
  - C3-hosny_score: C1 + the standardized 1-d zero-shot score.
  - hosny_zero_shot: the published log-odds used directly as the risk, with nothing fitted. It has no survival curve, so it has no IBS.
- Models alone: Pai and Hosny from the E3 PCA16 full-budget fits.
- **C4 stays with COLIPRI** (fixed in phase 0).
- Output goes to `outputs/lung1_ceiling_ladder/primary_tumour_centred`, so the phase 1 tables are not overwritten. The existing models are refitted with the same inner folds and bootstrap seed, which must reproduce the phase 1 numbers exactly as a check.
- New Holm families:
  - `tumour_centred_over_C1`: C3-pai / C3-hosny / C3-hosny_score vs C1.
  - `tumour_centred_vs_radiomics`: the same three vs C2.
  - `tumour_vs_whole_ct`: C3-pai, C3-hosny and C3-hosny_score vs C3-colipri; pai, hosny and hosny_zero_shot vs colipri.
- The phase 1 families are unchanged.

## Asymmetry (to state in the paper)

Radiomics, Pai and Hosny receive the tumour location (the GTV centroid). The whole-CT FMs do not.

## Deviation recorded on 21 October 2026: the Hosny reproduction check failed

The pre-specified check failed. Our zero-shot P(dead at 2 years) vs the authors' published Maastro predictions gave Spearman **0.765** (Pearson 0.33, median |difference| 0.007, n = 205). The published file has 5 rows without an id and duplicated ids; these were excluded. Seed distance to the Pai annotations: median 0.16 mm, max 3.6 mm. Following the rule, nothing downstream was run before this investigation.

**Investigation** (scripts in the session scratchpad; numbers reproduced here)

| Hypothesis | Test | Result |
| --- | --- | --- |
| Patient id mismatch in the published file | 2-year survival label of the published file vs our cohort | 100% agreement (205/205) |
| Axis order / flips | Exhaustive flip search against the 3 shipped input arrays | No flip, same orientation |
| Seed position (their contour ≠ TCIA GTV-1) | 1 mm shift search against the shipped arrays | Best shift 1–3 mm; best correlation 0.98 / 0.995 / **0.88** (LUNG1-006) |
| Model sensitivity to a 1–3 mm seed shift | Our score with the seed moved 1, 2, 3 mm in a random direction | Spearman with ourselves 0.976 / 0.963 / 0.963; with the published scores 0.76 / 0.78 / 0.74. A shift does not explain the gap. |
| Double interpolation (cache 1 mm → patch) | Patch built directly from the raw DICOM, in one step (linear / B-spline / nearest) | Spearman vs published 0.751 / 0.766 / 0.735; vs our cache-based scores 0.967 |
| Model port | Authors' 3 shipped arrays through our port | Published probabilities reproduced to 6 decimals |

**Conclusion.** The gap comes from the authors' input arrays, not from our pipeline or our port. Even after optimal alignment, their arrays differ from the TCIA CT (LUNG1-006: correlation 0.88). This points to a different source CT or contour on their side, which we cannot reproduce. The inter-reader Spearman that the authors report for their own model (0.88) also shows that the score is sensitive to the input.

**Discrimination.** On the same 205 patients and our 2-year labels:
- AUC 0.695 (published) vs 0.667 (ours).
- Paired bootstrap difference (2,000 resamples, seed 20261021): **−0.028 [−0.084, +0.025]**.
- Our AUC on all 415 patients with a known 2-year status is 0.603. The authors' test subset is easier than the full cohort.

**Decision.** We keep our preprocessing (the GTV centroid on the cached 1 mm CT) for every Hosny analysis: zero-shot, frozen embedding and fine-tuning.
- In the paper it is reported as a *reimplementation of Hosny et al. on the public TCIA data*, together with this failed check, the AUC comparison above and the published AUC on the 205-patient subset.
- The published scores cannot replace ours, because they cover only 205 of 416 patients and cannot be evaluated on our folds.
- Nothing else in the protocol changes.
