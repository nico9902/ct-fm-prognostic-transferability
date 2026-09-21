# NLST scan-level prognostic labels: investigation, design, and QC

Scope: build scan-level prognostic labels (one row per screening CT at
T0/T1/T2) for the `pilot_1000_balanced` imaging cohort (1000 patients, 3000
scans). **No model was trained and no COLIPRI embeddings were extracted** for
this task — only the label table below.

Script: [scripts/build_nlst_prognostic_labels.py](../scripts/build_nlst_prognostic_labels.py)
Output: `data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/`
(`nlst_scan_prognostic_labels.csv` / `.parquet`, `prognostic_label_qc.json`)

## 1. What NLST data is actually available locally

The local NLST clinical data was pulled via
[scripts/download_nlst_clinical.py](../scripts/download_nlst_clinical.py) from
the Imaging Data Commons (IDC) "bigquery_export" bucket. Listing that bucket
directly (`s5cmd --no-sign-request ls s3://idc-open-metadata/bigquery_export/idc_v24_clinical/`)
confirms IDC exposes exactly **five** NLST tables — there is no sixth table
to discover:

| Table | Rows (full) | Grain | Contents |
|---|---|---|---|
| `nlst_prsn` | 53,452 | 1 row / patient | demographics, screen results/isolation codes per round, **first-lung-cancer summary fields**, `scr_days0/1/2` |
| `nlst_canc` | 2,150 | 1 row / lung-cancer diagnosis (a patient can have >1, via `lc_order`) | staging, histology, `study_yr`, `candx_days`, `first_lc` |
| `nlst_screen` | 75,138 | 1 row / patient / study year | screen technical/quality fields |
| `nlst_ctab` | 177,487 | 1 row / observed CT abnormality | nodule/abnormality codes (used for the separate *diagnostic-state* labels, not prognosis) |
| `nlst_ctabc` | 31,046 | 1 row / abnormality comparison | prior-comparison codes |

**None of these five tables carries vital-status or mortality data** — no
death date, no cause of death, no "known alive as of" field. The full NLST
dataset held by NCI's CDAS does include a mortality table, but it is not part
of this repository's data and was not downloaded. This is a genuine data
availability gap, not an oversight in this script.

### Consequence for the requested endpoints

| Endpoint | Buildable locally? |
|---|---|
| 1. Future lung cancer diagnosis | **Yes** — from `nlst_prsn` (+ cross-checked against `nlst_canc`) |
| 2. Overall survival | **No** — no vital-status table available anywhere in the local/IDC NLST data |
| 3. Lung-cancer-specific survival | **No** — same reason; also cannot separate other-cause deaths |

Endpoints 2 and 3, and their fixed-horizon labels (`death_1y/3y/5y`,
`lung_cancer_death_3y/5y`, `os_event`, `os_time_days`,
`lung_cancer_death_event`, `lung_cancer_death_time_days`,
`other_cause_death`), are emitted in the output table **for schema
completeness only, as `NA` on every row**, each carrying the explicit flag
`mortality_endpoints_unavailable_no_local_nlst_death_table`. Nothing is
silently inferred or defaulted to "no event."

## 2. Variables used for the future-cancer-diagnosis endpoint

All from `nlst_prsn` (one row per patient), verified empirically against the
raw parquet (not assumed from memory of the data dictionary):

- **`scr_days0` / `scr_days1` / `scr_days2`**: days from randomization to the
  T0/T1/T2 screening exam. This is the per-scan landmark day
  (`scan_day_from_randomization`), selected per row by `timepoint_index`.
  Missing for 3/3000 scans in this cohort (all T2).
- **`candx_days`**: days from randomization to the participant's *first*
  confirmed lung cancer diagnosis. Verified against `nlst_canc`: for every
  patient, the `first_lc == 1` row's `candx_days` is identical to
  `nlst_prsn.candx_days` (0 mismatches across all 53,452 patients / 2,058
  cancer cases nationwide). 92 patients nationwide have additional, later
  primaries recorded in `nlst_canc` (`lc_order` 2–4); these are not modeled
  as separate events here, matching NLST's own "first cancer" convention in
  `nlst_prsn`.
- **`canc_free_days`**: populated for all patients (53,452/53,452), even
  cancer cases. Empirically, for every cancer case `canc_free_days <=
  candx_days` (never greater) — i.e. it is the day of the last assessment
  confirming the participant cancer-free, which can precede the eventual
  diagnosis day when there is a diagnostic workup lag. For non-cancer
  patients it is the day of last available follow-up (the natural censoring
  day).
- **`cancyr`, `can_scr`, `canc_rpt_link`**: preserved as raw metadata (which
  screening round is associated with the diagnosis, screen-detected vs.
  interval/off-schedule, and a linkage-quality flag). Their exact code
  definitions are not re-derived from an official data dictionary (not
  available locally) — they are carried through unmodified and are not used
  to compute any timing value, only kept for audit.

## 3. Landmarking rule (scan is the origin, not randomization)

For each scan, `scan_day_from_randomization` is the landmark. A scan is
**excluded** from the endpoint (`excluded_prior_cancer_diagnosis`) if the
first cancer was already diagnosed on or before that scan
(`scan_day >= candx_days`) — this also covers the one same-day
diagnosis-scan case found in this cohort. All exposed source values for an
excluded scan are still preserved in the row; only the derived label columns
are left `NA`.

For a scan that isn't excluded:
- If the participant is ever diagnosed **and** the scan precedes it:
  `cancer_event = 1`, `cancer_time_days = candx_days - scan_day` (always
  strictly positive by construction).
- Otherwise treat as censored, but **only if the assured follow-up actually
  covers this scan** (`scan_day <= canc_free_days`): `cancer_event = 0`,
  `cancer_time_days = canc_free_days - scan_day`.
  - If a non-cancer participant's `canc_free_days` predates the scan
    (`scan_day > canc_free_days`), the scan falls outside the assured
    follow-up window and cannot be honestly scored as censored (doing so
    would produce a negative time-to-censoring). These scans are left `NA`
    and flagged `scan_after_confirmed_followup_window` (1 case in this
    cohort). This was caught by an explicit non-negativity check during
    development — an earlier version of this script produced negative
    `cancer_time_days` for exactly this situation before the guard was
    added.

`cancer_censor_days` is populated **only for censored (event=0) rows**, where
it is identical to `cancer_time_days` (kept as a separate, self-describing
column per the requested schema, used by the horizon-label rule below). It
is intentionally left `NA` for event rows: `canc_free_days` can legitimately
predate the scan there (diagnostic-workup lag before the confirmed
diagnosis), which would otherwise expose a confusing negative auxiliary
value even though `cancer_time_days` itself is correct.

## 4. Fixed-horizon labels (`future_cancer_1y/2y/3y`)

For horizon *h* (365/730/1095 days):
- `1` if a diagnosis is confirmed within *h* days of the scan.
- `0` if either (a) a diagnosis is confirmed but strictly after *h*, or
  (b) the participant has confirmed cancer-free follow-up
  (`cancer_censor_days`) of at least *h* days.
- `NA` (unknown) otherwise — i.e. censored before *h* elapsed. **A scan is
  never labeled negative solely because no event was observed if the
  observed follow-up window is shorter than the horizon.**

## 5. Output schema

`nlst_scan_prognostic_labels.csv` / `.parquet`, 3000 rows × 33 columns:

- Identifiers: `patient_id`, `study_uid`, `series_uid`, `study_date`,
  `screening_round` (T0/T1/T2), `timepoint_index`, `scan_date_or_days`
  (= `study_date`; NLST DICOM dates are per-patient shifted for
  de-identification but preserve within-patient relative timing — the day-0
  anchor actually used for all time-to-event math is
  `scan_day_from_randomization`, not the calendar date).
- Preserved raw source fields: `cancyr`, `can_scr`, `canc_rpt_link`,
  `candx_days`, `canc_free_days`, `scr_days0/1/2`.
- Derived landmark: `scan_day_from_randomization`.
- Cancer diagnosis endpoint: `cancer_event`, `cancer_time_days`,
  `cancer_censor_days`, `future_cancer_1y/2y/3y`.
- Survival endpoints (always `NA`, see §1): `os_event`, `os_time_days`,
  `lung_cancer_death_event`, `lung_cancer_death_time_days`,
  `other_cause_death`, `death_1y/3y/5y`, `lung_cancer_death_3y/5y`.
- `prognostic_label_quality_flags`: semicolon-joined list, one of
  `missing_participant_link`, `prsn_canc_candx_days_mismatch`,
  `missing_scan_day_anchor`, `duplicate_patient_round`,
  `excluded_prior_cancer_diagnosis`, `scan_after_confirmed_followup_window`,
  `mortality_endpoints_unavailable_no_local_nlst_death_table` (present on
  every row), or nothing beyond the mortality flag if otherwise clean.

Note on a pre-existing column: the imaging manifest
(`nlst_pilot_1000_balanced_series.csv`) already carries a `cancer_label`
column, but it is constant per patient across all three rounds (500
cancer / 500 non-cancer, by cohort construction) and therefore carries no
landmark timing information — using it directly as a "future cancer" label
at T0/T1 would be look-ahead leakage for patients ultimately diagnosed after
that scan (or would give a positive label to a scan that occurs after the
diagnosis, which is not "future" cancer at all). It is not reused here; the
`cancer_event`/`future_cancer_*y` columns above are the landmark-safe
replacement.

## 6. QC statistics

Cohort: 1000 patients, 3000 scans (1000 T0 + 1000 T1 + 1000 T2).

- **Excluded — cancer already diagnosed at/before the scan**: 1 scan (1
  patient, T2), `excluded_prior_cancer_diagnosis`.
- **Excluded — missing landmark day** (`scr_days2` absent): 3 scans (3
  patients, all T2), `missing_scan_day_anchor`.
- **Excluded — scan outside assured follow-up window**: 1 scan (1 patient,
  T2), `scan_after_confirmed_followup_window`.
- Total not eligible for the cancer endpoint: 5/3000 scans.

### Cancer diagnosis endpoint

| | N |
|---|---|
| Eligible scans | 2995 |
| Eligible patients | 1000 |
| Events (`cancer_event=1`) | 1496 |
| Censored (`cancer_event=0`) | 1499 |

By round: events T0=500, T1=500, T2=496; censored T0=500, T1=500, T2=499
(the 5 T2 exclusions/missing-anchor account for the shortfall from 1000).

Follow-up / time-to-event distribution from the scan (days), eligible scans:

| | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| `cancer_time_days` (event+censored combined) | 2995 | 1560.7 | 691.2 | 0 | 1047.5 | 1691 | 2109.5 | 2838 |
| `cancer_censor_days` (censored only) | 1499 | 2022.1 | 398.7 | 0 | 1750 | 2055 | 2330.5 | 2838 |

### Fixed-horizon labels

| Horizon | Positive | Negative | Unknown (censored before horizon) | Not eligible |
|---|---|---|---|---|
| 1y | 172 | 2817 | 6 | 5 |
| 2y | 423 | 2556 | 16 | 5 |
| 3y | 760 | 2205 | 30 | 5 |

Prevalence among labeled (positive / (positive+negative)): 1y = 5.75%,
2y = 14.20%, 3y = 25.63%. These reflect the cohort's deliberate 50/50
cancer/non-cancer enrichment (see `cancer_label` note above), not the
population base rate.

### Survival endpoints

Not computed (see §1). All rows: `os_event`, `os_time_days`,
`lung_cancer_death_event`, `lung_cancer_death_time_days`,
`other_cause_death`, `death_1y/3y/5y`, `lung_cancer_death_3y/5y` = `NA`,
flagged `mortality_endpoints_unavailable_no_local_nlst_death_table`.

Full machine-readable QC: `prognostic_label_qc.json` alongside the label
table.

## 7. Recommendation

To build endpoints 2 and 3 (overall survival, lung-cancer-specific
survival), the NLST mortality/vital-status table is required from NCI's CDAS
(the full NLST dataset, not the IDC imaging-linked subset used here). Until
that table is obtained, only the future-lung-cancer-diagnosis endpoint above
is usable for evaluating frozen COLIPRI representations.
