#!/usr/bin/env python3
"""Build and audit scan-level current-diagnostic labels for an NLST cohort.

The labels are deliberately separate from cancer outcomes.  Binary labels use
the nullable convention: 1=positive, 0=negative, blank=unknown/not linkable.
The SCT abnormality tables are one-row-per-observed-abnormality tables; after a
valid screen/round match, absence of a code is therefore treated as negative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AB_CODES = {
    "nodule_ge_4mm": "51",
    "micronodule_lt_4mm": "52",
    "emphysema": "59",
    "consolidation": "58",
    "fibrosis_or_scarring": "61",
    "adenopathy_or_mass": "56",
}
SIGNIFICANT_CODES = {str(x) for x in range(51, 65)}
RECOMMENDATION_COLUMNS = {
    "diagnostic_ct_recommended": ["rec_diag_ct"],
    "short_interval_ct_recommended": [
        "rec_ld_or_thin_ct_3mo", "rec_ld_or_thin_ct_3_6mo", "rec_ld_or_thin_ct_6mo"
    ],
    "pet_recommended": ["rec_fdg_pet"],
    "biopsy_recommended": ["rec_biopsy"],
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--clinical-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dicom-root", type=Path, default=None)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def clean_code(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def nullable_binary(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").astype("Int8")


def aggregate_abnormalities(ctab: pd.DataFrame) -> pd.DataFrame:
    ctab = ctab.copy()
    ctab["pid_key"] = ctab["dicom_patient_id"].astype("string")
    ctab["study_yr_key"] = clean_code(ctab["study_yr"])
    ctab["ab_code_clean"] = clean_code(ctab["sct_ab_desc"])
    key = ["pid_key", "study_yr_key"]
    rows = []
    for k, g in ctab.groupby(key, dropna=False, sort=False):
        codes = g["ab_code_clean"].dropna().tolist()
        rows.append({
            "pid_key": k[0], "study_yr_key": k[1],
            "ctab_row_count": len(g),
            "ctab_codes_raw": ";".join(codes),
            "ctab_codes_unique": ";".join(sorted(set(codes))),
            "ctab_has_significant_record": int(bool(set(codes) & SIGNIFICANT_CODES)),
            **{name: int(code in set(codes)) for name, code in AB_CODES.items()},
        })
    return pd.DataFrame(rows)


def raw_comparison(ctabc: pd.DataFrame) -> pd.DataFrame:
    x = ctabc.copy()
    x["pid_key"] = x["dicom_patient_id"].astype("string")
    x["study_yr_key"] = clean_code(x["study_yr"])
    x["sct_ab_code_clean"] = clean_code(x["sct_ab_code"])
    return (x.groupby(["pid_key", "study_yr_key"], dropna=False, sort=False)
            .agg(ctabc_row_count=("sct_ab_code_clean", "size"),
                 ctabc_codes_raw=("sct_ab_code_clean", lambda s: ";".join(s.dropna())))
            .reset_index())


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    series = read(a.series)
    prsn = read(a.clinical_dir / "nlst_prsn.csv")
    screen = read(a.clinical_dir / "nlst_screen.csv")
    ctab = read(a.clinical_dir / "nlst_ctab.csv")
    ctabc = read(a.clinical_dir / "nlst_ctabc.csv")
    canc = read(a.clinical_dir / "nlst_canc.csv")

    # The manifest's round is accepted only when its integer index is the same
    # as the clinical study_yr key.  No date-nearest or row-order fallback.
    out = series.copy()
    out["patient_id"] = out["patient_id"].astype("string")
    out["study_yr"] = clean_code(out["timepoint_index"])
    out["timepoint"] = out["timepoint"].where(out["timepoint"].isin(["T0", "T1", "T2"]))
    out["series_path"] = ""
    if a.dicom_root:
        out["series_path"] = out.apply(
            lambda r: str(a.dicom_root / "nlst" / str(r.patient_id) / str(r.study_uid)
                        / f"CT_{r.series_uid}"), axis=1)

    # Participant is one row per patient. Keep raw participant screening fields.
    prsn["patient_id"] = prsn["dicom_patient_id"].astype("string")
    participant_cols = [c for c in prsn.columns if c not in {"dicom_patient_id", "pid", "patient_id"}]
    p = prsn[["patient_id"] + participant_cols].copy()
    for k in range(3):
        for c in [f"scr_iso{k}", f"scr_res{k}", f"scr_days{k}"]:
            if c in p.columns:
                p[c] = p[c].astype("string")
    out = out.merge(p, on="patient_id", how="left", validate="many_to_one", indicator="participant_merge")

    # Wide participant values become the current-round raw values.
    for base in ["scr_iso", "scr_res", "scr_days"]:
        out[f"{base}_raw"] = out.apply(
            lambda r: r.get(f"{base}{r.get('study_yr', '')}", pd.NA), axis=1).astype("string")

    s = screen.copy()
    s["pid_key"] = s["dicom_patient_id"].astype("string")
    s["study_yr_key"] = clean_code(s["study_yr"])
    screen_keys = ["pid_key", "study_yr_key"]
    # Preserve all available screen-level raw fields, including quality fields.
    screen_payload = s.drop(columns=["dicom_patient_id", "pid", "study_yr", "dataset_version"], errors="ignore")
    out["pid_key"] = out["patient_id"]
    out["study_yr_key"] = out["study_yr"]
    out = out.merge(screen_payload, on=screen_keys, how="left", validate="one_to_one", indicator="screen_merge")
    ab = aggregate_abnormalities(ctab)
    out = out.merge(ab, on=screen_keys, how="left", validate="one_to_one")
    out = out.merge(raw_comparison(ctabc), on=screen_keys, how="left", validate="one_to_one")

    screen_valid_codes = {str(x) for x in range(1, 7)}
    iso = clean_code(out["scr_iso_raw"])
    res = clean_code(out["scr_res_raw"])
    screen_linked = out["screen_merge"].eq("both")
    out["isolation_suspicious"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out.loc[screen_linked & iso.isin(screen_valid_codes), "isolation_suspicious"] = iso.eq("4").astype("Int8")
    out["comparison_suspicious"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out.loc[screen_linked & res.isin(screen_valid_codes), "comparison_suspicious"] = res.isin({"4", "5", "6"}).astype("Int8")

    # A missing ctab row means no observed abnormality because this is the
    # official one-record-per-observed-abnormality table. It is still unknown
    # when the corresponding screen/round cannot be linked.
    ctab_ok = screen_linked
    for name in [*AB_CODES, "any_significant_abnormality"]:
        if name not in out.columns:
            out[name] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    for name in AB_CODES:
        out.loc[ctab_ok, name] = out.loc[ctab_ok, name].fillna(0).astype("Int8")
    out.loc[ctab_ok, "any_significant_abnormality"] = out.loc[ctab_ok, "ctab_has_significant_record"].fillna(0).astype("Int8")

    # Recommendations are intentionally unavailable in this IDC export.
    available_recs = []
    for name, candidates in RECOMMENDATION_COLUMNS.items():
        cols = [c for c in candidates if c in out.columns]
        if cols:
            available_recs.append(name)
            out[name] = out[cols].apply(pd.to_numeric, errors="coerce").max(axis=1).astype("Int8")
        else:
            out[name] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out["escalated_followup"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    if available_recs:
        out["escalated_followup"] = out[available_recs].max(axis=1, skipna=True).astype("Int8")

    # Cancer timing/outcome fields are copied as outcome metadata only.
    canc_cols = [c for c in ["cancyr", "can_scr", "canc_rpt_link", "candx_days", "canc_free_days"] if c in prsn.columns]
    # They already came from prsn; make their role explicit in the metadata.
    out["linkage_qc"] = "ok"
    out.loc[out["participant_merge"] != "both", "linkage_qc"] = "missing_participant"
    out.loc[out["timepoint"].isna(), "linkage_qc"] = "invalid_manifest_round"
    out.loc[out["screen_merge"] != "both", "linkage_qc"] = "missing_screen_round"
    out.loc[out.duplicated(["patient_id", "timepoint"], keep=False), "linkage_qc"] = "duplicate_patient_round"
    out["linkage_qc_ok"] = (out["linkage_qc"] == "ok").astype("Int8")

    # Independent chronology audit of the manifest assignment.  A duplicated
    # date is ambiguous even if the existing manifest supplies T0/T1/T2.
    def chronology_status(g: pd.DataFrame) -> str:
        dates = pd.to_datetime(g["study_date"], errors="coerce")
        day_values = pd.to_numeric(g["scr_days_raw"], errors="coerce").dropna().tolist()
        if dates.notna().sum() != len(g) or not dates.is_monotonic_increasing:
            return "date_missing_or_nonmonotonic"
        if dates.duplicated().any():
            return "ambiguous_duplicate_date"
        if day_values != sorted(day_values) or len(day_values) != len(set(day_values)):
            return "scr_days_nonmonotonic_or_duplicate"
        return "ok"
    chron = out.groupby("patient_id", sort=False).apply(chronology_status, include_groups=False).rename("chronology_qc")
    out = out.merge(chron, on="patient_id", how="left", validate="many_to_one")
    out.loc[out["chronology_qc"] != "ok", "linkage_qc"] = "chronology_ambiguous"
    out["linkage_qc_ok"] = (out["linkage_qc"] == "ok").astype("Int8")

    # Do not expose labels for a round whose assignment is ambiguous.  The
    # row and raw values remain in the canonical file for audit/reconciliation.
    label_names = ["isolation_suspicious", "comparison_suspicious", *AB_CODES,
                   "any_significant_abnormality", "diagnostic_ct_recommended",
                   "short_interval_ct_recommended", "pet_recommended",
                   "biopsy_recommended", "escalated_followup"]
    for label in label_names:
        out.loc[out["linkage_qc"] != "ok", label] = pd.NA

    # Drop internal merge keys but retain all source raw columns.
    out = out.drop(columns=["pid_key", "study_yr_key", "participant_merge", "screen_merge"], errors="ignore")
    out.to_csv(a.output_dir / "nlst_scan_diagnostic_labels.csv", index=False)
    try:
        out.to_parquet(a.output_dir / "nlst_scan_diagnostic_labels.parquet", index=False)
    except ImportError:
        pass

    labels = ["isolation_suspicious", "comparison_suspicious", *AB_CODES, "any_significant_abnormality",
              "diagnostic_ct_recommended", "short_interval_ct_recommended", "pet_recommended",
              "biopsy_recommended", "escalated_followup"]
    summary = []
    for label in labels:
        for round_name, g in [("ALL", out), ("T0", out[out.timepoint == "T0"]), ("T1", out[out.timepoint == "T1"]), ("T2", out[out.timepoint == "T2"])]:
            v = pd.to_numeric(g[label], errors="coerce")
            summary.append({"label": label, "timepoint": round_name, "n_total": len(g),
                            "n_valid": int(v.notna().sum()), "n_missing": int(v.isna().sum()),
                            "n_positive": int((v == 1).sum()), "n_negative": int((v == 0).sum()),
                            "prevalence": float((v == 1).sum() / v.notna().sum()) if v.notna().any() else None})
    pd.DataFrame(summary).to_csv(a.output_dir / "label_summary.csv", index=False)
    for x, y in [("isolation_suspicious", "comparison_suspicious"), ("comparison_suspicious", "nodule_ge_4mm"), ("comparison_suspicious", "emphysema")]:
        pd.crosstab(out[x].fillna(-1), out[y].fillna(-1), margins=True).to_csv(a.output_dir / f"crosstab_{x}_vs_{y}.csv")

    qc = {"n_series": len(out), "n_patients": int(out.patient_id.nunique()),
          "duplicate_series_uid": int(out.series_uid.duplicated().sum()),
          "duplicate_patient_round": int(out.duplicated(["patient_id", "timepoint"]).sum()),
          "linkage_qc_counts": out.linkage_qc.value_counts(dropna=False).to_dict(),
          "chronology_qc_counts": out[["patient_id", "chronology_qc"]].drop_duplicates().chronology_qc.value_counts(dropna=False).to_dict(),
          "screen_round_counts": {"|".join(map(str, k)): int(v) for k, v in out.groupby(["timepoint", "linkage_qc"], dropna=False).size().items()},
          "recommendation_fields_available": available_recs,
          "cancer_outcome_fields": canc_cols,
          "abnormality_code_mapping": {**AB_CODES, "significant": sorted(SIGNIFICANT_CODES)}}
    (a.output_dir / "linkage_qc.json").write_text(json.dumps(qc, indent=2, default=str))
    print(json.dumps(qc, indent=2, default=str))


if __name__ == "__main__":
    main()
