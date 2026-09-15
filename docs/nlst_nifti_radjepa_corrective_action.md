# NLST NIfTI corrective action for RadJEPA-3D

Date: 2026-09-15

## Objective

Make the `pilot_1460_balanced` NIfTI cohort safe to use as the source for
RadJEPA-3D preprocessing. Source DICOMs and existing NIfTIs are preserved. New
repairs and RadJEPA tensors must be written to versioned output directories.

## Audit performed

The following source manifest was audited:

```text
/ssd2/domenico/datasets/nlst/processed/nlst/pilot_1460_balanced/nii_manifest.csv
```

The audit opened all 4,108 NIfTI headers and the DICOM headers for every
selected series. It checked:

- NIfTI existence, readability, dimensionality, affine and voxel spacing;
- DICOM modality and localizer image type;
- diagnostic slice count and anatomical coverage;
- constant in-plane shape and orientation;
- unique physical slice positions and inter-slice gaps;
- agreement between DICOM slice spacing and the NIfTI header;
- agreement between the number of physical planes and NIfTI depth;
- the orientation that will result from lossless canonicalization.

Thresholds used were at least 50 diagnostic planes, at least 200 mm coverage,
an axial slice-normal component of at least 0.9, a maximum gap of 1.5 times the
median gap, and 5% spacing tolerance.

## Audit result

| Result | Scans | Patients |
|---|---:|---:|
| Ready without rebuilding | 3,952 | 1,448 |
| Excluded at first pass | 156 | 118 |
| Patients with at least two ready scans | — | 1,416 |

Exclusion reasons overlap for a small number of scans:

| Reason | Scans |
|---|---:|
| Repeated physical slice position | 119 |
| Large slice gap | 27 |
| Fewer than 50 diagnostic slices | 12 |
| Inconsistent orientation inside one series | 2 |
| Less than 200 mm coverage | 1 |

The 119 repeated-position series were inspected at the pixel level:

- 97 contain identical duplicate planes and can be repaired safely by keeping
  one copy at each position;
- 22 contain different pixel data at the same positions and must not be
  deduplicated automatically.

For the 59 scans that cannot be repaired by exact deduplication, the IDC index
contains a metadata-eligible alternate axial series for 32 study rounds. The
other 27 study rounds have no alternate series with at least 50 images in IDC
and must remain excluded unless another source becomes available.

## Corrections implemented

### 1. Fixed the original series-selection defect

`scripts/build_nlst_manifest.py` previously returned `-1e8 + image_count` for
series below the minimum count and then retained scores greater than `-1e8`.
This admitted every positive image count, including one-image series. Rejected
series now receive `-1e9` and cannot pass the cutoff.

### 2. Added exhaustive pre-RadJEPA validation

`scripts/validate_nlst_nifti_for_radjepa.py` creates:

- a row-level report for all scans;
- a ready-only NIfTI manifest;
- a JSON summary of thresholds and exclusions.

It is read-only with respect to DICOM and NIfTI sources. Validation is parallel
over series and records orientation, spacing, coverage, gaps, localizers and
failure reasons.

### 3. Made DICOM conversion refuse unsafe geometry

`scripts/convert_nlst_dicom_to_nifti.py` now:

- removes DICOM images explicitly marked as localizers;
- removes repeated planes only when decoded pixels and HU scaling are equal;
- refuses conflicting images at the same physical position;
- enforces minimum slice count and anatomical coverage;
- refuses large slice gaps;
- records every applied repair in the output warning column.

Existing files are not overwritten unless `--overwrite` is explicitly passed.
Repairs must use a new output directory.

### 4. Canonicalized RadJEPA input generation

`scripts/prepare_radjepa_volumes.py` now converts each valid NIfTI to canonical
RAS array orientation before constructing `(C,D,H,W)`. Canonicalization uses
the NIfTI affine and only permutes/flips axes; it does not interpolate the
source NIfTI.

The output manifest now records source orientation, prepared orientation,
source spacing, effective post-resize spacing and preprocessing version. The
`--require-qc-ready` option prevents an unvalidated manifest from being used.
Existing `.npy` files are marked `existing_unverified`, because their format
does not contain enough provenance to certify canonicalization.

### 5. Added recovery utilities

- `scripts/download_idc_series.py` downloads an explicit UID list.
- `scripts/merge_nlst_nifti_manifests.py` combines ready, repaired and
  replacement conversion manifests while rejecting duplicate study/timepoint
  rows.
- `scripts/run_server_validate_prepare_radjepa_pilot_max_balanced.sh` runs the
  validation and creates versioned canonical RadJEPA inputs.

## Generated audit artifacts

The repository contains the reviewable outputs under:

```text
outputs/preprocessing_qc/pilot_1460_balanced/
├── nifti_qc_report.csv
├── nifti_qc_summary.json
├── nii_manifest_radjepa_ready.csv
├── duplicate_slice_classification.csv
├── repair_identical_duplicates_series.csv
├── replacement_candidates_from_idc.csv
├── replacement_series_manifest.csv
└── replacement_series_uids.txt
```

## Data repair procedure

Set paths once:

```bash
export DATASET_ROOT=/ssd2/domenico/datasets/nlst
export TAG=pilot_1460_balanced
export QC_REPO_DIR=outputs/preprocessing_qc/pilot_1460_balanced
export REPAIR_DIR="${DATASET_ROOT}/processed/nlst/${TAG}/nifti_repair_v2"
```

### A. Rebuild the 97 exact-duplicate series

```bash
.venv/bin/python scripts/convert_nlst_dicom_to_nifti.py \
  --manifest "${QC_REPO_DIR}/repair_identical_duplicates_series.csv" \
  --dicom-root "${DATASET_ROOT}/raw/nlst/dicom" \
  --output-root "${REPAIR_DIR}/nii_exact_duplicates" \
  --output-manifest "${REPAIR_DIR}/nii_exact_duplicates_manifest.csv" \
  --overwrite \
  --on-error skip
```

This operation writes new NIfTIs and leaves the original 97 files unchanged.

### B. Download and convert the 32 candidate replacement series

```bash
.venv/bin/python scripts/download_idc_series.py \
  --series-uids "${QC_REPO_DIR}/replacement_series_uids.txt" \
  --download-dir "${DATASET_ROOT}/raw/nlst/dicom"

.venv/bin/python scripts/convert_nlst_dicom_to_nifti.py \
  --manifest "${QC_REPO_DIR}/replacement_series_manifest.csv" \
  --dicom-root "${DATASET_ROOT}/raw/nlst/dicom" \
  --output-root "${REPAIR_DIR}/nii_replacements" \
  --output-manifest "${REPAIR_DIR}/nii_replacements_manifest.csv" \
  --overwrite \
  --on-error skip
```

IDC metadata eligibility is only a first filter. Every downloaded replacement
must pass the same DICOM geometry validator after conversion.

### C. Merge and validate the repaired cohort

```bash
.venv/bin/python scripts/merge_nlst_nifti_manifests.py \
  --base-ready-manifest "${QC_REPO_DIR}/nii_manifest_radjepa_ready.csv" \
  --additional-manifest "${REPAIR_DIR}/nii_exact_duplicates_manifest.csv" \
  --additional-manifest "${REPAIR_DIR}/nii_replacements_manifest.csv" \
  --output "${REPAIR_DIR}/nii_manifest_candidate_v2.csv"

.venv/bin/python scripts/validate_nlst_nifti_for_radjepa.py \
  --nifti-manifest "${REPAIR_DIR}/nii_manifest_candidate_v2.csv" \
  --output-ready-manifest "${REPAIR_DIR}/nii_manifest_radjepa_ready_v2.csv" \
  --output-qc-report "${REPAIR_DIR}/nifti_qc_report_v2.csv" \
  --output-summary "${REPAIR_DIR}/nifti_qc_summary_v2.json"
```

The second validation is mandatory. Its ready manifest is the authoritative
input to RadJEPA preprocessing.

### D. Generate new RadJEPA arrays

```bash
.venv/bin/python scripts/prepare_radjepa_volumes.py \
  --nifti-manifest "${REPAIR_DIR}/nii_manifest_radjepa_ready_v2.csv" \
  --output-root "${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_npy_v2" \
  --output-manifest "${DATASET_ROOT}/processed/nlst/${TAG}/radjepa_npy_manifest_v2.csv" \
  --require-qc-ready
```

Do not reuse the old `radjepa_npy` directory: those arrays were generated
before geometry filtering and canonical orientation were introduced.

## Expected final scope

If all 97 deterministic repairs and all 32 replacement candidates pass, the
maximum recoverable cohort is 4,081 scan timepoints. At least 27 timepoints
remain unavailable because IDC has no metadata-eligible alternative in the
same study. The final number may be lower if a replacement fails DICOM-level
geometry QC.

The clean cohort should therefore be defined by the final ready manifest,
rather than by an assumption that every originally selected timepoint is
usable.
