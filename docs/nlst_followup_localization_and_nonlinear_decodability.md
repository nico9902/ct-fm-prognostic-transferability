# NLST follow-up: localization audit and nonlinear probes

## Executive result

Experiment 1 was not run as a supervised window classification experiment.
The available NLST tables do not provide a consistently linkable nodule
location for the prepared NPY volumes. `nlst_ctab.csv` contains
`sct_slice_num`, but it is populated only for code 51 (nodule >=4 mm), has
duplicate rows and invalid/sentinel values, and does not consistently refer to
the slice numbering of the selected DICOM series. A reliable experiment would
require annotations tied to the selected series, preferably DICOM
`InstanceNumber`/SOP Instance UID or `ImagePositionPatient` z-coordinate (plus
the annotation's physical coordinate or slice range).

Experiment 2 was run on 2,984 scans from 999 patients using the existing
frozen Rad-JEPA cache and existing patient-level split. The small MLP did not
produce a meaningful improvement for either `isolation_suspicious` or
`nodule_ge_4mm`; both remain effectively chance-level. This supports the
interpretation that the negative result is not merely a failure of linear
access, at least within this probe family and capacity.

## Experiment 1: localization metadata audit

The relevant available fields are:

| Source | Fields | Assessment |
|---|---|---|
| `nlst_ctab.csv` | `sct_ab_desc`, `sct_slice_num`, nodule diameters, margins, location descriptors | Only candidate nodule slice field; no series UID, SOP UID, z-coordinate, centroid, or slice range |
| selected-series manifest | patient/study/series UIDs, image count | Reliably links the clinical round to the selected series, but does not add nodule coordinates |
| DICOM series | `InstanceNumber`, `ImagePositionPatient`, orientation and spacing | Available for the selected volumes, and used to order/prepare slices, but no nodule annotation is linked to these fields |
| prepared NPY manifest | native depth, source shape, foreground bounds | Preserves native z length; the preparation does not crop z, but cannot resolve an annotation whose source slice identity is ambiguous |

Audit findings on the 1,000-patient/3-round clinical export:

- 8,056 CTAB rows are present; 2,014 have a non-null `sct_slice_num`.
- Every localized row has `sct_ab_desc == 51`; there are no slice locations for
  the other abnormality codes.
- The field has 231 distinct values and 27 duplicate records at the same
  patient/round/series/slice key.
- After linking by patient and round to the selected series, 2,014 rows have a
  series match (2,011 on prepared volumes); values include sentinel `999` and values larger than
  the selected series depth. Representative contradictions include
  patient 103621/T2: slice 239 vs 191 DICOM/NPY slices; 103639/T2: 314 vs
  164; 133231/T1: 503 vs 165; and 119958/T0: 999 vs 164.

Thus `sct_slice_num` cannot safely be converted to an NPY index using either
`slice_num - 1` or reverse ordering (`depth - slice_num`). The value may refer
to a different source reconstruction or source-series numbering. The
available `sct_epi_loc` is a categorical anatomic descriptor, not a z
coordinate.

To enable window labels, obtain one of the following for each nodule and
selected series: a series UID plus DICOM InstanceNumber/SOP Instance UID;
physical z-coordinate/3-D centroid in the selected series coordinate system;
or an annotation slice range with the source-series identity and a validated
mapping to the selected series. Then a 32-slice window can be positive when
its native slice interval intersects the nodule's annotated slice interval.

## Experiment 2: linear versus nonlinear probes

The probe used the same 2,984-scan usable cohort, fixed patient split (599
train, 200 validation, 200 test patients), training-only standardization and
class weighting, five seeds, and validation early stopping. Rad-JEPA and all
cached representations were frozen. The MLP was
`input -> Linear(128) -> ReLU -> Dropout(0.2) -> Linear(1)`.

AUROC is mean across seeds; delta is MLP minus linear.

| Target | Representation | Linear | MLP | Delta |
|---|---|---:|---:|---:|
| isolation_suspicious | resize32 | 0.505 | 0.506 | +0.001 |
| isolation_suspicious | sliding_mean | 0.507 | 0.500 | -0.007 |
| isolation_suspicious | sliding_max | 0.499 | 0.505 | +0.007 |
| isolation_suspicious | sliding_meanmax | 0.505 | 0.505 | +0.001 |
| nodule_ge_4mm | resize32 | 0.502 | 0.503 | +0.001 |
| nodule_ge_4mm | sliding_mean | 0.504 | 0.501 | -0.003 |
| nodule_ge_4mm | sliding_max | 0.497 | 0.509 | +0.012 |
| nodule_ge_4mm | sliding_meanmax | 0.503 | 0.510 | +0.006 |
| emphysema | resize32 | 0.540 | 0.552 | +0.012 |
| emphysema | sliding_mean | 0.532 | 0.549 | +0.018 |
| emphysema | sliding_max | 0.555 | 0.545 | -0.010 |
| emphysema | sliding_meanmax | 0.551 | 0.536 | -0.014 |
| fibrosis_or_scarring | resize32 | 0.553 | 0.531 | -0.023 |
| fibrosis_or_scarring | sliding_mean | 0.554 | 0.543 | -0.011 |
| fibrosis_or_scarring | sliding_max | 0.548 | 0.545 | -0.003 |
| fibrosis_or_scarring | sliding_meanmax | 0.542 | 0.565 | +0.022 |

The MLP's small nodule improvement on sliding-max and sliding-meanmax is not
consistent evidence of decodability: the mean AUROC remains 0.509 and 0.510,
respectively, with no meaningful separation from chance. Isolation is also
unchanged. Weak emphysema/fibrosis signals persist, but nonlinear access does
not provide a general improvement.

## Reproducibility artifacts

- [nonlinear probe script](../scripts/run_diagnostic_nonlinear_decodability.py)
- `outputs/diagnostic_nonlinear_decodability/per_seed_results.csv`
- `outputs/diagnostic_nonlinear_decodability/summary_results.csv`
- `outputs/diagnostic_nonlinear_decodability/metadata.json`
- Existing cache and split source: `/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability/`
