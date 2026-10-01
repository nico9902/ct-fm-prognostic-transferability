# Verified correction: patient 215070 has T1 and T2, not T0 and T1

This verification supersedes the provisional interpretation below. All **324 local DICOM headers** were checked. The 156-slice axial series has StudyDate `20000102`, ClinicalTrialTimePointID `T1`, and SeriesDescription screening year `1`. The 167-slice axial series has StudyDate `20010102`, ClinicalTrialTimePointID `T2`, and SeriesDescription screening year `2`. The one-image scout also belongs to T2.

[TCIA's official NLST CT documentation](https://wiki.cancerimagingarchive.net/display/NLST/NLST%2BCT%2BImages%2B--%2BGeneral%2BDetails) states that dates are fictitious screening-year codes: `19990102` = year 0, `20000102` = year 1, `20010102` = year 2. The first SeriesDescription field is the screening year. Consequently, the two diagnostic volumes are **T1 and T2**, and **T0 is not locally available**. The generated manifest incorrectly renumbered available studies from zero, including the scout as an independent screening timepoint.

The isolated COLIPRI cache metadata and NPZ round identifiers have been corrected by series UID. Clinical diagnostic and prognostic labels were rebuilt with the existing scripts from `verified_series_manifest.csv`. Both diagnostic rows pass linkage QC, the chronology audit passes, and both prognostic rows have valid screening-day anchors (463 and 729 days). This yields one verified T1→T2 imaging pair, separated by **266 clinical days**. The fictitious DICOM date difference must not be used as the clinical interval.

The patient is now recovered with verified round linkage and labels in the isolated recovery directory. The completed gate cohorts, feature caches and patient splits remain unchanged; adding the patient to a future analysis requires an explicitly recorded split assignment, since it was absent from the existing splits.

Evidence: `verified_dicom_timepoints.csv`, `timepoint_verification.json`, `verified_series_manifest.csv`, `verified_diagnostic_labels/`, `verified_prognostic_labels/`, and corrected `colipri/*/cache/embedding_metadata.csv`.

---

## Earlier provisional recovery notes (superseded)

# Patient 215070 preprocessing recovery attempt

The selected T1 series was retried using `scripts/prepare_radjepa_sliding_volume.py` with unchanged defaults (50 minimum slices, 200 mm minimum coverage, window normalization). It again failed because its only DICOM image is explicitly tagged LOCALIZER. It cannot form a diagnostic CT volume.

All three locally available patient series were inspected. The 167-slice axial series currently assigned T2 shares the scout date (2001-01-02), and its description starts with `2,OPA`, consistent with a candidate second screening. The manifest builder orders distinct StudyInstanceUIDs by date and UID, so separate same-day studies occupy separate timepoint positions. This explains the duplicate-date chronology flag; it does not independently validate the clinical screening-round linkage.

The axial series was successfully reprocessed in an isolated recovery directory as **T1_candidate**, producing a finite float32 volume of shape **1 × 167 × 256 × 256**, normalized to [0,1]. The result exactly matches the existing T2 prepared array. Thus this is a provisional timepoint correction candidate, not an additional scan recovered from the failed scout. Only two local diagnostic CT volumes are present; no distinct third diagnostic screening volume was identified.

The original manifests, clinical labels, feature caches, patient splits and completed gate results were not changed. Reinstatement requires verification of the candidate's clinical screening-round linkage, rebuilding its appropriate labels and extracting COLIPRI features. A third diagnostic scan must not be fabricated by using this volume twice.

Artifacts: `t1_retry_input.csv`, `t1_retry_result.csv`, `local_series_inventory.json`, `axial_candidate_input.csv`, `axial_candidate_result.csv`, `validation.json`, and the isolated `sliding_npy/` volume.

## Frozen COLIPRI feature recovery

Both available diagnostic CTs now have verified frozen COLIPRI caches in the isolated `colipri/` directory. Mean and max embeddings each have shape 2 × 864, attention embeddings 2 × 768, and pre-pooling grids per scan 864 × 24 × 24 × 24. All embeddings are finite. The exact prior-gate checkpoint hash and model revision were reused; official attention-pooling equivalence passed. See `feature_recovery_validation.json` and `extract_recovered_features.py`.

Clinical evidence was inspected and saved as `nlst_prsn_patient_evidence.csv` and `nlst_screen_patient_evidence.csv`. Participant screening days are 1, 463 and 729, and clinical screen rows exist for study_yr 0, 1 and 2. Those records do not contain a direct SeriesInstanceUID link verifying the provisional axial candidate. DICOM dates are 2000-01-02 and 2001-01-02, with the scout sharing the second date. Round linkage remains unverified. No clinical target was attached to the candidate, and no patient split was assigned or changed. The patient is recovered at the imaging-feature level, but remains excluded from clinical gate analysis pending verified round linkage.

## T0 retrieval search

Queried the full IDC NLST CT index for PatientID 215070 and independently queried TCIA's public getSeries endpoint for the same patient. Both returned the exact same five SeriesInstanceUIDs: T1 axial CT (156 images), T1 scout (1 image), T2 axial CT (167 images), and two T2 scouts (one image each). No screening-year-0 series is available in either queried catalog. The clinical T0 screening record exists, but it does not establish that T0 imaging was released. Consequently T0 cannot currently be downloaded or recovered from these public sources. This does not prove that the original clinical image never existed.

Search evidence: `idc_full_patient_series.csv`, `tcia_series_query_response.txt`, `tcia_series_query_status.json`, and `t0_recovery_search.json`. Existing T1/T2 volumes, caches, labels, gate cohorts and splits remain unchanged.
