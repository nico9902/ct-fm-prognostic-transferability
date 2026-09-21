#!/usr/bin/env python3
"""Build scan-level prognostic labels (future cancer diagnosis) for an NLST cohort.

Scope and known data gap
-------------------------
The locally available NLST clinical data was downloaded from the Imaging Data
Commons (IDC) "bigquery_export" bucket, which exposes exactly five NLST
tables: nlst_prsn, nlst_canc, nlst_screen, nlst_ctab, nlst_ctabc (verified by
listing s3://idc-open-metadata/bigquery_export/idc_v24_clinical/ directly).
None of these tables carries vital-status/mortality fields (no death date, no
cause of death, no follow-up-alive indicator). The full NLST dataset held by
NCI's CDAS does contain a mortality table, but it is not present in this
repository's data. Consequently this script can only build the "future lung
cancer diagnosis" endpoint. The overall-survival and lung-cancer-specific
survival columns are emitted for schema completeness but are always NA, and
every row is flagged accordingly rather than silently omitted.

Landmark and source variables used (all from nlst_prsn, one row per patient)
------------------------------------------------------------------------------
- scr_days0 / scr_days1 / scr_days2: days from randomization to the T0/T1/T2
  screening exam. Used as the per-scan landmark day (verified 1:1 against the
  imaging manifest's patient/round pairing).
- candx_days: days from randomization to the participant's FIRST confirmed
  lung cancer diagnosis (verified against nlst_canc: for every patient, the
  row with first_lc == 1 has an identical candx_days value; some patients
  have additional, later primaries in nlst_canc that are not separately
  modeled here, matching NLST's own single "first cancer" convention in
  nlst_prsn).
- canc_free_days: days the participant is known to have been free of a
  cancer diagnosis. Empirically, for every cancer case candx_days >=
  canc_free_days (never less), i.e. it is the last cancer-free assessment
  day at or before diagnosis; for non-cases it is the day of last available
  follow-up (censoring day). It is populated for all 53,452 participants in
  the raw table.
- cancyr / can_scr / canc_rpt_link: preserved as raw metadata about how/when
  the first cancer was detected (screen round, screen-detected vs interval,
  linkage flag). Not required to compute time-to-event and not otherwise
  reinterpreted here beyond what is documented in the report.

Landmarking rule
-----------------
For each scan, the day of that scan (scan_day) is scr_days{round}. A scan is
excluded from the cancer endpoint (excluded_prior_cancer_diagnosis) if the
participant's first cancer was already diagnosed on or before that scan
(scan_day >= candx_days) -- this also covers same-day diagnostic scans. For
all other scans:
  - if the participant is ever diagnosed (candx_days not null) and the scan
    precedes it: cancer_event=1, cancer_time_days = candx_days - scan_day.
  - otherwise: cancer_event=0 (censored), cancer_time_days = canc_free_days -
    scan_day.
cancer_censor_days = canc_free_days - scan_day is populated for every
eligible scan regardless of event status: it is the duration from this scan
during which the participant is positively confirmed cancer-free, and is
what fixed-horizon labels are checked against for cases that are censored
before the horizon.

Fixed-horizon binary labels (future_cancer_{1,2,3}y) follow the rule: label=1
if a future diagnosis is confirmed within the horizon; label=0 if either (a)
diagnosis is confirmed but strictly after the horizon, or (b) the
participant has confirmed cancer-free follow-up (cancer_censor_days) at
least as long as the horizon; otherwise label=NA (unknown -- censored before
the horizon elapsed). A scan is never labeled negative solely because no
event was observed if the observed follow-up window is shorter than the
horizon.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HORIZONS_DAYS = {"1y": 365, "2y": 730, "3y": 1095}
PRSN_RAW_COLS = [
    "cancyr", "can_scr", "canc_rpt_link", "candx_days", "canc_free_days",
    "scr_days0", "scr_days1", "scr_days2",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", type=Path, required=True, help="Scan-level imaging manifest (one row per patient/round/series).")
    p.add_argument("--clinical-dir", type=Path, required=True, help="Directory with nlst_prsn.csv and nlst_canc.csv for the cohort.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def build(series: pd.DataFrame, prsn: pd.DataFrame, canc: pd.DataFrame) -> pd.DataFrame:
    out = series[["patient_id", "study_uid", "series_uid", "study_date", "timepoint", "timepoint_index"]].copy()
    out = out.rename(columns={"timepoint": "screening_round"})
    out["patient_id"] = out["patient_id"].astype("string")
    out["scan_date_or_days"] = out["study_date"]

    p = prsn.copy()
    p["patient_id"] = p["dicom_patient_id"].astype("string")
    p = p[["patient_id"] + [c for c in PRSN_RAW_COLS if c in p.columns]]
    for c in ["candx_days", "canc_free_days", "scr_days0", "scr_days1", "scr_days2"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    out = out.merge(p, on="patient_id", how="left", validate="many_to_one", indicator="participant_merge")

    def scan_day_for(row) -> float:
        col = f"scr_days{row['timepoint_index']}"
        return row.get(col, pd.NA)

    out["scan_day_from_randomization"] = out.apply(scan_day_for, axis=1)
    out["scan_day_from_randomization"] = pd.to_numeric(out["scan_day_from_randomization"], errors="coerce")

    # Cross-check: prsn's first-cancer fields should match nlst_canc's
    # first_lc==1 rows. Deviations are surfaced as a QC flag, not silently
    # trusted.
    first = canc.copy()
    first["patient_id"] = first["dicom_patient_id"].astype("string")
    first["candx_days"] = pd.to_numeric(first["candx_days"], errors="coerce")
    first = first[first["first_lc"].astype(str) == "1"][["patient_id", "candx_days"]].rename(
        columns={"candx_days": "candx_days_canc_first_lc"}
    )
    out = out.merge(first, on="patient_id", how="left", validate="many_to_one")
    out["candx_days_consistency_ok"] = (
        out["candx_days"].isna() & out["candx_days_canc_first_lc"].isna()
    ) | (out["candx_days"] == out["candx_days_canc_first_lc"])

    flags = pd.Series([[] for _ in range(len(out))], index=out.index)

    def add_flag(mask, name):
        for i in out.index[mask]:
            flags.loc[i].append(name)

    add_flag(out["participant_merge"] != "both", "missing_participant_link")
    add_flag(~out["candx_days_consistency_ok"].fillna(True), "prsn_canc_candx_days_mismatch")
    add_flag(out["scan_day_from_randomization"].isna(), "missing_scan_day_anchor")
    add_flag(out.duplicated(["patient_id", "screening_round"], keep=False), "duplicate_patient_round")

    has_cancer = out["candx_days"].notna()
    has_anchor = out["scan_day_from_randomization"].notna()
    excluded_prior = has_cancer & has_anchor & (out["scan_day_from_randomization"] >= out["candx_days"])
    add_flag(excluded_prior, "excluded_prior_cancer_diagnosis")

    # A non-cancer participant's canc_free_days is their last confirmed
    # cancer-free assessment; if it falls before this scan, the scan sits
    # outside the assured follow-up window and cannot be scored as censored
    # (that would imply a negative, meaningless time-to-censoring).
    censored_candidate = ~has_cancer & has_anchor & out["canc_free_days"].notna()
    followup_covers_scan = out["scan_day_from_randomization"] <= out["canc_free_days"]
    scan_after_followup_window = censored_candidate & ~followup_covers_scan
    add_flag(scan_after_followup_window, "scan_after_confirmed_followup_window")

    event_mask = has_cancer & has_anchor & ~excluded_prior
    censored_mask = censored_candidate & followup_covers_scan

    out["cancer_event"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out["cancer_time_days"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["cancer_censor_days"] = pd.Series(pd.NA, index=out.index, dtype="Float64")

    out.loc[event_mask, "cancer_event"] = 1
    out.loc[event_mask, "cancer_time_days"] = out.loc[event_mask, "candx_days"] - out.loc[event_mask, "scan_day_from_randomization"]
    out.loc[censored_mask, "cancer_event"] = 0
    out.loc[censored_mask, "cancer_time_days"] = out.loc[censored_mask, "canc_free_days"] - out.loc[censored_mask, "scan_day_from_randomization"]
    # cancer_censor_days is only meaningful for censored rows (it is then
    # identical to cancer_time_days, kept as a distinct, self-describing
    # column per the requested schema). For event rows the event day itself
    # already fully characterizes the timing, and canc_free_days can predate
    # the scan (diagnostic workup lag), which would otherwise render a
    # confusing negative value here.
    out.loc[censored_mask, "cancer_censor_days"] = out.loc[censored_mask, "canc_free_days"] - out.loc[censored_mask, "scan_day_from_randomization"]
    eligible = event_mask | censored_mask

    for name, horizon in HORIZONS_DAYS.items():
        col = f"future_cancer_{name}"
        lab = pd.Series(pd.NA, index=out.index, dtype="Int8")
        confirmed_within = event_mask & (out["cancer_time_days"] <= horizon)
        confirmed_after = event_mask & (out["cancer_time_days"] > horizon)
        confirmed_negative_censored = censored_mask & (out["cancer_censor_days"] >= horizon)
        lab.loc[confirmed_within] = 1
        lab.loc[confirmed_after | confirmed_negative_censored] = 0
        out[col] = lab

    # Overall survival and lung-cancer-specific survival: no vital-status /
    # mortality source table exists in the locally available NLST clinical
    # data (see module docstring). Emitted as NA for schema completeness.
    for c in [
        "os_event", "os_time_days",
        "lung_cancer_death_event", "lung_cancer_death_time_days", "other_cause_death",
        "death_1y", "death_3y", "death_5y",
        "lung_cancer_death_3y", "lung_cancer_death_5y",
    ]:
        out[c] = pd.NA
    add_flag(pd.Series(True, index=out.index), "mortality_endpoints_unavailable_no_local_nlst_death_table")

    out["prognostic_label_quality_flags"] = flags.apply(lambda fs: ";".join(fs) if fs else "ok")

    out = out.drop(columns=["participant_merge", "candx_days_consistency_ok", "candx_days_canc_first_lc"], errors="ignore")
    return out


def qc_stats(out: pd.DataFrame) -> dict:
    stats: dict = {}
    stats["n_scans_total"] = len(out)
    stats["n_patients_total"] = int(out["patient_id"].nunique())
    stats["n_scans_by_round"] = out["screening_round"].value_counts(dropna=False).to_dict()
    stats["n_excluded_prior_cancer_diagnosis"] = int(
        out["prognostic_label_quality_flags"].str.contains("excluded_prior_cancer_diagnosis").sum()
    )
    stats["n_missing_scan_day_anchor"] = int(
        out["prognostic_label_quality_flags"].str.contains("missing_scan_day_anchor").sum()
    )

    cancer_eligible = out["cancer_event"].notna()
    stats["cancer_endpoint"] = {
        "n_eligible_scans": int(cancer_eligible.sum()),
        "n_eligible_patients": int(out.loc[cancer_eligible, "patient_id"].nunique()),
        "n_events": int((out["cancer_event"] == 1).sum()),
        "n_censored": int((out["cancer_event"] == 0).sum()),
        "n_events_by_round": out.loc[out["cancer_event"] == 1, "screening_round"].value_counts().to_dict(),
        "n_censored_by_round": out.loc[out["cancer_event"] == 0, "screening_round"].value_counts().to_dict(),
        "cancer_time_days_distribution": out.loc[cancer_eligible, "cancer_time_days"].describe().to_dict(),
        "cancer_censor_days_distribution": out.loc[cancer_eligible, "cancer_censor_days"].describe().to_dict(),
    }

    for name in HORIZONS_DAYS:
        col = f"future_cancer_{name}"
        v = out[col]
        n_valid = int(v.notna().sum())
        stats[col] = {
            "n_positive": int((v == 1).sum()),
            "n_negative": int((v == 0).sum()),
            "n_unknown_insufficient_followup": int(cancer_eligible.sum() - n_valid) if cancer_eligible.sum() >= n_valid else None,
            "n_not_eligible": int((~cancer_eligible).sum()),
            "prevalence_among_labeled": float((v == 1).sum() / n_valid) if n_valid else None,
        }

    stats["mortality_endpoints"] = {
        "status": "unavailable",
        "reason": (
            "No vital-status/death table exists among the locally available NLST "
            "clinical tables sourced from IDC (nlst_prsn, nlst_canc, nlst_screen, "
            "nlst_ctab, nlst_ctabc). Verified by listing "
            "s3://idc-open-metadata/bigquery_export/idc_v24_clinical/ which exposes "
            "only these five nlst_* tables. os_event, os_time_days, "
            "lung_cancer_death_event, lung_cancer_death_time_days, other_cause_death, "
            "death_1y/3y/5y and lung_cancer_death_3y/5y are emitted as NA for all rows."
        ),
    }
    return stats


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    series = read(a.series)
    prsn = read(a.clinical_dir / "nlst_prsn.csv")
    canc = read(a.clinical_dir / "nlst_canc.csv")

    out = build(series, prsn, canc)
    out.to_csv(a.output_dir / "nlst_scan_prognostic_labels.csv", index=False)
    try:
        out.to_parquet(a.output_dir / "nlst_scan_prognostic_labels.parquet", index=False)
    except ImportError:
        pass

    stats = qc_stats(out)
    (a.output_dir / "prognostic_label_qc.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
