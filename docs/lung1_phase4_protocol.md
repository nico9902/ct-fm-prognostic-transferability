# Phase 4: external validation on NSCLC-Radiogenomics, pre-specified protocol

**Fixed on:** 21 October 2026. Written before any Radiogenomics image was processed and before any model was applied to Radiogenomics. Amendments are allowed only before the frozen models are applied (step 4.4). Each amendment is dated and appended at the end.  
**Training cohort:** the 416 LUNG1 patients of phases 1–2.  
**Test cohort:** NSCLC-Radiogenomics (TCIA, version 4 of 1 June 2021).

## Choices agreed with the PI (21 October 2026)

1. **Population.**
   - The primary population is the patients with a tumour segmentation (DICOM SEG, n = 144). All models can be computed on them and every comparison is paired.
   - The secondary population is all patients with an eligible CT and survival (up to 211). It is used only for the models that do not need the tumour: C0 and the five whole-CT FMs alone.
2. **Time origin:** the date of the CT used as input. The prediction is made "from the image". In LUNG1, the planning CT precedes the start of radiotherapy only by a short time.
3. **Horizon:** the primary metric is Uno C at 36 months, as in LUNG1. 60 months is a sensitivity.

## Data

- **Clinical:** `data/clinical/radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv` (TCIA, 211 patients, sha256 `0a1f7cdf…1071476`). Dates are shifted by TCIA consistently within each patient.
- **Survival:**
  - `os_event` = 1 if Survival Status is Dead.
  - `os_time_days` = (date of death if dead, otherwise date of last known alive) − CT date.
  - The CT date is the DICOM StudyDate of the selected CT series. Any disagreement with the CSV "CT Date" is reported.
  - Patients with a non-positive time are excluded.
- **CT series:**
  - For the 144 patients with a SEG, the series is the one the SEG references.
  - For the others, the rule is fixed before seeing any image. The candidates are axial series, not a localizer, with at least 50 slices, and whose description does not indicate a scout, fusion, PET or attenuation-correction CT, MIP, or coronal/sagittal reformat. Among them, the series with the most slices is taken, with ties broken by the lowest series number. If there is more than one study, the one whose date is closest to the CSV CT Date is used.
- **Tumour mask:** the single segment of the SEG.
  - TCIA labels it "Heart", "Tissue" or "Segmentation"; this is a known labelling defect, and it is the tumour.
  - **Check:** the mask centroid must lie within 5 mm of the seed in the Pai repository, and the mask must overlap the R231 lung mask dilated by 10 voxels. A patient who fails is excluded from the primary population and listed.

## Covariates (mapping to LUNG1 coding)

| LUNG1 variable | Radiogenomics source | Mapping |
| --- | --- | --- |
| age | Age at Histological Diagnosis | as is |
| sex | Gender | lower-case |
| T | Pathological T stage | Tis → 1, T1a/T1b → 1, T2a/T2b → 2, T3 → 3, T4 → 4 |
| N | Pathological N stage | N0/N1/N2 → 0/1/2 |
| stage | derived from pTNM, AJCC 7th edition | I (incl. 0/Tis), II, IIIa, IIIb; stage IV (M1, n = 4) → IIIb, the highest LUNG1 category |
| histology | Histology | Adenocarcinoma → adenocarcinoma; Squamous cell carcinoma → squamous; NSCLC NOS → nos |
| GTV volume | the SEG mask on the 1 mm grid | same as LUNG1 |

LUNG1 uses *clinical* TNM and Radiogenomics uses *pathological* TNM. The mapping is identical, and the difference is a stated limitation.

## Image processing

The processing is identical to LUNG1, with no change to any parameter:
- **Five FMs:** `original_ct` input, each with its official preprocessing (`scripts/extract_lung1_e1_embeddings.py`).
- **Tumour-based methods:** the cached 1 mm thorax crop (`thorax_hu`), with the GTV centroid as seed. These are Pai, Hosny and the PyRadiomics Original features (same settings as LUNG1).
- No harmonization of any kind (no ComBat, no kernel or intensity matching). Differences in kernel and slice thickness are described, not corrected.

## Frozen models (4.3)

Each model is refitted on all 416 LUNG1 patients with **the same selection procedure** used internally:
- penalties from the same grid by 3-fold event-stratified CV on the 416 patients;
- imputation, scaling and PCA fitted on the 416.

Every fitted object is saved together with the hashes of its inputs. This happens **before** any Radiogenomics survival is read by an analysis script.

| Model | Content |
| --- | --- |
| C0, C1, C2, C3-*fm* (5 FMs), C4-colipri, C4-tangerine | ladder of phase 1 (PCA16 image blocks) |
| C3-pai, C3-hosny_score | ladder of phase 2 |
| *fm* alone (5 FMs), pai alone, hosny alone | linear Cox, PCA16 (E3 full budget) |
| Hosny zero-shot | published log-odds, no fit |
| stage + volume, radiomics | E3 baselines |

The following are **not** carried over: LoRA, partial, MLP, SQA and fine-tuned Hosny. All of them were worse than or equal to the frozen linear model on LUNG1.

## Application and analysis (4.4)

The models are applied **once**. The risk is the linear predictor (for Hosny zero-shot, the log-odds).

- **Primary metric.** Uno C at 36 months. The censoring distribution is estimated on the Radiogenomics cohort, because the model is fixed.
- **Intervals.** Patient bootstrap: 2,000 resamples, seed 20261021, paired across models. They are percentile intervals; the p-values use a Wald z with the bootstrap SE.
- **Primary contrasts** (one family, Holm):
  1. COLIPRI alone − stage + volume
  2. C3-colipri − C1
  3. COLIPRI alone − Pai alone
  4. COLIPRI alone − Hosny zero-shot
- **Chance family** (Holm): every model vs 0.5.
- **Secondary** (descriptive, with intervals):
  - all other models and contrasts;
  - Uno C at 60 months;
  - Harrell C;
  - calibration slope (Cox regression of the outcome on the linear predictor) and observed/expected events at 36 months, using the LUNG1 Breslow baseline;
  - Spearman correlation between the internal LUNG1 C and the external C across models (does the ranking transport?).
- **Secondary population** (all eligible CTs): C0 and the five FMs alone, plus COLIPRI − C0.
- **Power.** It is stated in advance that, with about 34 events by 36 months, intervals of about ±0.08–0.10 are expected for each C. The minimum detectable difference between two models is expected to be ≥ 0.08. Non-significant contrasts are therefore not evidence of equivalence, and TOST with ±0.03 is reported only for completeness. The minimum detectable difference is reported from the bootstrap SE.
- **Pre-registered hypothesis.** The Radiogenomics patients are early-stage surgical patients, so many deaths may not be cancer-related. If the FM signal is age, habitus and anatomy (phase 1.3), the FMs should transport *at least as well as* stage + volume, which varies little here.
- **No change after the results:** no new model, no other horizon, no other series selection.

## Amendments

**Amendment 1 (21 October 2026, after the header-only cohort audit, before any image was decoded or any outcome analysed): CT series rule for patients without a SEG.** The first rule took the series with the most slices. Its description filter excluded a valid "ChestScreen" series, did not catch attenuation-correction CTs ("CTAC") or whole-body PET/CT CTs ("CT WB"), and preferred abdomen/pelvis series because they have more slices. The new rule:
- excludes scout/topogram/localizer, fusion, PET, CTAC/attenuation, whole-body, MIP, coronal/sagittal, abdomen/pelvis ("abd", "pel", "AP") and delayed series;
- puts descriptions with chest/thorax/lung first, then slice thickness ≤ 3 mm, then the most slices, with the lowest series number breaking ties.

The 144 SEG-referenced series are unaffected.

Resulting cohort (`data/derived/radiogenomics/cohort.csv`):
- 211 patients.
- 10 have no eligible CT: 8 with only PET/CT fusion CTs, 1 with only an attenuation-correction CT, 1 with only whole-body CTs. One more (AMC-046) has its CT after the last contact.
- **Secondary population: 200** (60 deaths, 42 by 36 months).
- **Primary population: 144** (51 deaths, 34 by 36 months, median follow-up 3.8 years); before the GTV checks.
- For two SEG-referenced patients, the DICOM CT date differs from the CSV "CT Date" by more than a year (R01-106: −401 days; R01-120: +731 days). By protocol, the DICOM date is the time origin.
- The 49 AMC patients have no pathological TNM ("Not Collected"). **In the secondary population, C0 and COLIPRI − C0 are therefore computed on the patients with pTNM only** (all the R01 patients). The five FMs alone use all 200.

**Amendment 2 (21 October 2026, after decoding one CT, before any model input was computed): out-of-field padding.**
- Some GE series store out-of-field voxels as PixelPaddingValue, which decodes to −2048 HU (22% of the voxels in R01-122).
- In LUNG1, the minimum decoded value is exactly −1024 HU in all 416 patients.
- For every Radiogenomics input (the five FMs, the thorax cache, Pai, Hosny and radiomics), voxels below −1024 HU are set to −1024 at read time. This makes the decoded CT equivalent to LUNG1's. It is a DICOM decoding fix, not an intensity harmonization: nothing above −1024 HU changes.

**Amendment 3 (21 October 2026, during the first image pass, before any embedding was analysed or any model applied): slice-level CT audit.**
- The DICOM reader warned about missing slices in one series. The cohort builder now applies the same slice audit that LUNG1 used (`build_lung1_cohort.scan_ct`), with the same fatal issues: header id mismatch or read error, duplicate SOP UID, inconsistent geometry, duplicate slice position, and gap ratio > 1.5.
- A SEG-referenced series that fails excludes the patient: R01-119 (slice gap).
- For patients without a SEG, the first candidate in the priority order that passes is taken. AMC-006 moves from "LUNG 1 MM" (gap) to "ST CAP 5MM".
- All 199 remaining series have a gap ratio of 1.0.
- **Primary population: 143. Secondary population: 199.**
- The partial embeddings computed before this amendment were deleted and recomputed.
