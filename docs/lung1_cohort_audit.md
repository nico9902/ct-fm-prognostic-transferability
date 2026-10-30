# LUNG1 cohort audit for prognostic transferability

## Source and outcome definition

The imaging inventory is the local TCIA NSCLC-Radiomics version 4 DICOM download, whose `metadata.csv` lists one CT series, one RTSTRUCT series, and usually one SEG series per subject. The clinical source is TCIA's `NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv`, copied to `data/clinical/lung1/NSCLC_Radiomics_Lung1.clinical-version3-Oct_2019.csv` on 2026-10-01. Its SHA-256 is `132f72b58b9660bf5e6b24b9817b335f1896360bc253e1f2034a3ffee593e6fd`.

TCIA states that `Survival.time` is days from treatment start and `deadstatus.event` is 1 for death and 0 for right censoring. TCIA describes the CTs as pretreatment, but the clinical CSV does not include a treatment start date, so scan timing cannot be checked independently. [TCIA NSCLC-Radiomics collection](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics)

## Pre-model selection rules for the primary analysis

1. Join the clinical and imaging inventories by exact `PatientID` / `Subject ID`. Require one clinical row and one CT series per patient. The release has exactly one CT series per subject, so no scan-ranking choice is needed.
2. Require positive OS time and event code 0 or 1. Missing clinical covariates do not exclude a patient from the imaging-only analysis.
3. Require the downloaded CT file count to equal TCIA metadata and all CT DICOM headers to match the selected patient and series. Exclude a CT from the strict whole-volume primary cohort if its geometry is inconsistent, any slice position is duplicated, or the largest interslice step is more than 1.5 times the median positive step. Keep the excluded scans identifiable for a later sensitivity analysis.
4. For GTV-based analyses additionally require a matching RTSTRUCT containing a `GTV-1` ROI and a usable matching DICOM SEG primary-neoplasm mask. Compute volume from positive SEG voxels and voxel spacing. The RTSTRUCT is the source that identifies the GTV-1 contour; TCIA says its primary-tumor ROI is represented in SEG.
5. Apply these patient-level rules before any foundation-model extraction. Model-specific preprocessing failures must be reported separately, with the common patient intersection used for paired comparisons.

## Current header-audit result

| Population | Patients | Death events | Censored |
| --- | ---: | ---: | ---: |
| Source clinical and imaging join | 422 | 373 | 49 |
| Strict global CT cohort | 417 | 369 | 48 |
| GTV cohort after mask-volume audit | 416 | 369 | 47 |

The five strict CT exclusions are `LUNG1-014`, `LUNG1-021`, `LUNG1-085`, `LUNG1-194`, and `LUNG1-246`. The first three are already documented by TCIA as having missing slices. `LUNG1-194` has a doubled interslice step; `LUNG1-246` has a duplicate slice position. `LUNG1-128` passes the global CT rule but lacks a `GTV-1` RTSTRUCT ROI and a SEG object, so it is excluded from the GTV cohort. The five CT cases remain in the audit file with explicit reasons.

Among the 417 global CT candidates, 22 lack age, 39 lack histology, and one lacks overall stage. All have OS fields. Clinical baseline models must define their own missing-data handling inside training folds; these gaps do not affect the imaging-only denominator.

All 416 regional candidates produced a nonempty primary-neoplasm SEG mask with consistent frame spacing. Their voxel-derived GTV volumes range from 0.50 to 661.35 cm³ (median 39.08 cm³). These volumes are suitable for the initial cohort table; the smallest and largest masks should receive visual review during image preprocessing.

## Artifacts and reproduction

- `data/derived/lung1/cohort_audit.csv`: one row per patient with survival fields, clinical covariates, selected series UIDs and paths, observed CT file count, geometry gap ratio, ROI availability, eligibility statuses, and issue codes.
- `data/derived/lung1/cohort_with_gtv_volume.csv`: same rows with voxel-derived GTV volumes and any mask-volume failures.
- `scripts/build_lung1_cohort.py`: reproduces the join and full DICOM-header audit.
- `scripts/add_lung1_gtv_volume.py`: streams only primary-neoplasm SEG frames to compute GTV volume in cm³.

Run:

```bash
.venv/bin/python scripts/build_lung1_cohort.py
.venv/bin/python scripts/add_lung1_gtv_volume.py
```

The clinical CSV is linked from the [TCIA collection's “Lung1 clinical” download](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics). The raw download is required locally before running the scripts.

The `data/` directory is gitignored. The two scripts and this report are versionable; the raw TCIA CSV and patient-level derived tables remain local. The header pass checks identifiers, geometry, and file counts. It does not decode every CT pixel or visually inspect image quality and tumor-mask alignment. These checks are required during preprocessing before the final cross-model common cohort is declared.
