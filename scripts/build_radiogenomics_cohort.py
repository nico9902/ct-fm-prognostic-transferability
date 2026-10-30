#!/usr/bin/env python3
"""NSCLC-Radiogenomics external cohort for phase 4, in the LUNG1 cohort format.

Pre-specified in docs/lung1_phase4_protocol.md. Reads DICOM headers only (one file per
series) and the TCIA clinical CSV; no image is decoded.

* CT series: the series referenced by the SEG when there is one; otherwise (amendment 1
  of the protocol) the axial, non-localizer series with >= 50 slices whose description is
  not a scout, fusion, PET/attenuation-correction or whole-body CT, MIP, coronal/sagittal
  reformat, abdomen/pelvis or delayed series, from the study closest to the CSV CT Date;
  chest/thorax/lung descriptions first, then slice thickness <= 3 mm, then most slices
  (ties: lowest series number).
* Survival from the StudyDate of the selected CT to death or last known alive.
* Covariates mapped to the LUNG1 coding (pathological TNM, AJCC 7th stage groups).

Status columns: region_status = "ready" for the primary population (SEG + CT + survival),
global_status = "ready" for every patient with an eligible CT and survival.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from build_lung1_cohort import scan_ct

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/ssd2/domenico/datasets/NSCLC-Radiogenomics/dicom/NSCLC_Radiogenomics-6-1-21-Version-4")
CLINICAL = ROOT / "data/clinical/radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
OUTPUT = ROOT / "data/derived/radiogenomics"
# Fatal slice-level issues, identical to build_lung1_cohort.py (protocol amendment 3).
FATAL = {"ct_header_id_mismatch", "ct_header_read_error", "duplicate_ct_sop_uid", "inconsistent_ct_geometry",
         "duplicate_slice_position", "ct_slice_gap"}
EXCLUDE = re.compile(r"scout|topo|locali|fusion|pet|ctac|\bac\b|atten|mip|\bcor|sag|\bwb\b|whole.?body|"
                     r"abd|pel|\bap\b|delay", re.I)
CHEST = re.compile(r"chest|thorax|lung", re.I)


def header(folder: Path) -> tuple[pydicom.Dataset, int]:
    files = sorted(folder.glob("*.dcm"))
    return pydicom.dcmread(files[0], stop_before_pixels=True), len(files)


def series_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in metadata.loc[metadata.Modality == "CT"].iterrows():
        folder = BASE / r["File Location"].lstrip("./")
        ds, n = header(folder)
        orient = np.asarray(getattr(ds, "ImageOrientationPatient", [np.nan] * 6), dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        rows.append({"patient_id": r["Subject ID"], "ct_series_uid": r["Series UID"],
                     "ct_study_uid": r["Study UID"], "ct_series_dir": str(folder),
                     "description": str(r["Series Description"]), "n_files": n,
                     "series_number": int(getattr(ds, "SeriesNumber", 0) or 0),
                     "study_date": pd.to_datetime(str(getattr(ds, "StudyDate", "")), format="%Y%m%d", errors="coerce"),
                     "axial": bool(np.isfinite(normal).all() and abs(normal[2]) > 0.9),
                     "localizer": "LOCALIZER" in [str(v).upper() for v in getattr(ds, "ImageType", [])],
                     "slice_thickness_mm": float(getattr(ds, "SliceThickness", np.nan) or np.nan),
                     "kernel": str(getattr(ds, "ConvolutionKernel", "")),
                     "manufacturer": str(getattr(ds, "Manufacturer", "")),
                     "model": str(getattr(ds, "ManufacturerModelName", ""))})
    return pd.DataFrame(rows)


def seg_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in metadata.loc[metadata.Modality == "SEG"].iterrows():
        folder = BASE / r["File Location"].lstrip("./")
        files = sorted(folder.glob("*.dcm"))
        if len(files) != 1:
            raise ValueError(f"{r['Subject ID']}: expected one SEG file")
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        rows.append({"patient_id": r["Subject ID"], "seg_series_uid": r["Series UID"], "seg_file": str(files[0]),
                     "seg_referenced_series_uid": ds.ReferencedSeriesSequence[0].SeriesInstanceUID,
                     "seg_label": str(ds.SegmentSequence[0].SegmentLabel), "seg_segments": len(ds.SegmentSequence)})
    return pd.DataFrame(rows)


def stage_group(t: str, n: str, m: str) -> str:
    """AJCC 7th edition stage group, collapsed to the LUNG1 categories."""
    if m.startswith("M1"):
        return "IIIb"  # stage IV -> highest LUNG1 category (protocol)
    if n == "N3" or (t == "T4" and n == "N2"):
        return "IIIb"
    if n == "N2" or (t == "T3" and n == "N1") or (t == "T4" and n in ("N0", "N1")):
        return "IIIa"
    if (t in ("T1a", "T1b", "T2a") and n == "N1") or (t == "T2b" and n in ("N0", "N1")) or (t == "T3" and n == "N0"):
        return "II"
    if t in ("Tis", "T1a", "T1b", "T2a") and n == "N0":
        return "I"
    raise ValueError(f"Unmapped pTNM {t} {n} {m}")


def clinical(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    date = lambda c: pd.to_datetime(d[c], format="%m/%d/%Y", errors="coerce")
    t_map = {"Tis": 1, "T1a": 1, "T1b": 1, "T2a": 2, "T2b": 2, "T3": 3, "T4": 4}
    n_map = {"N0": 0, "N1": 1, "N2": 2}
    histology = {"Adenocarcinoma": "adenocarcinoma", "Squamous cell carcinoma": "squamous cell carcinoma",
                 "NSCLC NOS (not otherwise specified)": "nos"}
    collected = d["Pathological T stage"].ne("Not Collected") & d["Pathological N stage"].ne("Not Collected")
    out = pd.DataFrame({
        "patient_id": d["Case ID"], "affiliation": d["Patient affiliation"],
        "age": pd.to_numeric(d["Age at Histological Diagnosis"], errors="coerce"),
        "sex": d["Gender"].str.lower(),
        "t_stage": d["Pathological T stage"].map(t_map), "n_stage": d["Pathological N stage"].map(n_map),
        "m_stage": d["Pathological M stage"].map(lambda v: 1 if str(v).startswith("M1") else (0 if v == "M0" else np.nan)),
        "ptnm": d["Pathological T stage"] + " " + d["Pathological N stage"] + " " + d["Pathological M stage"],
        "overall_stage": [stage_group(t, n, m) if ok else np.nan for t, n, m, ok in
                          zip(d["Pathological T stage"], d["Pathological N stage"], d["Pathological M stage"], collected)],
        "histology": d["Histology "].map(histology),
        "os_event": d["Survival Status"].map({"Dead": 1, "Alive": 0}),
        "end_date": date("Date of Death").where(d["Survival Status"].eq("Dead"), date("Date of Last Known Alive")),
        "csv_ct_date": date("CT Date"),
    })
    if out.histology.isna().any() or out.os_event.isna().any() or out.end_date.isna().any():
        raise ValueError("Unmapped histology, survival status or end date")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metadata = pd.read_csv(BASE / "metadata.csv")
    series, segs, clin = series_table(metadata), seg_table(metadata), clinical(CLINICAL)
    if segs.patient_id.duplicated().any():
        raise ValueError("More than one SEG per patient")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.output_dir / "ct_series_audit.csv", index=False)
    audits = {}

    def audit(row) -> tuple[list[str], float | None]:
        if row.ct_series_uid not in audits:
            audits[row.ct_series_uid] = scan_ct(sorted(Path(row.ct_series_dir).glob("*.dcm")), row.patient_id,
                                                row.ct_series_uid)
        return audits[row.ct_series_uid]

    chosen = []
    for pid, group in series.groupby("patient_id"):
        seg = segs.loc[segs.patient_id == pid]
        if len(seg):
            pick = group.loc[group.ct_series_uid == seg.seg_referenced_series_uid.iloc[0]]
            rule = "seg_reference"
            if len(pick) == 1 and FATAL & set(audit(pick.iloc[0])[0]):
                chosen.append({"patient_id": pid, "selection_rule": rule,
                               "issues": "SEG-referenced CT fails slice audit: " + ",".join(audit(pick.iloc[0])[0])})
                continue
        else:
            ok = group.loc[group.axial & ~group.localizer & (group.n_files >= 50)
                           & ~group.description.str.contains(EXCLUDE)]
            csv_date = clin.loc[clin.patient_id == pid, "csv_ct_date"]
            if len(ok) and len(csv_date) and pd.notna(csv_date.iloc[0]):
                gap = (ok.study_date - csv_date.iloc[0]).abs()
                ok = ok.loc[gap == gap.min()]
            ok = ok.assign(tier=~ok.description.str.contains(CHEST),  # chest/thorax/lung first
                           thick=~(ok.slice_thickness_mm <= 3.0))  # then slices <= 3 mm
            ranked = ok.sort_values(["tier", "thick", "n_files", "series_number"], ascending=[True, True, False, True])
            passing = [i for i, row in ranked.iterrows() if not FATAL & set(audit(row)[0])]
            pick = ranked.loc[passing[:1]]  # first candidate passing the LUNG1 slice audit
            rule = "header_rule"
        if len(pick) != 1:
            chosen.append({"patient_id": pid, "selection_rule": rule, "issues": "no eligible CT series"})
            continue
        issues, gap = audit(pick.iloc[0])
        chosen.append({**pick.iloc[0].to_dict(), "selection_rule": rule, "ct_max_gap_ratio": gap,
                       "ct_audit_issues": ",".join(issues), "issues": ""})
    chosen = pd.DataFrame(chosen)
    data = clin.merge(chosen, on="patient_id", how="left").merge(segs, on="patient_id", how="left")
    data["os_time_days"] = (data.end_date - data.study_date).dt.days.astype(float)
    data["ct_date_minus_csv_days"] = (data.study_date - data.csv_ct_date).dt.days
    data["ct_expected_slices"] = data["ct_found_slices"] = data.n_files.astype("Int64")
    data["has_seg"] = data.seg_file.notna()
    eligible = data.ct_series_uid.notna() & (data.os_time_days > 0)
    data.loc[data.ct_series_uid.notna() & ~(data.os_time_days > 0), "issues"] = "non-positive survival time"
    data.loc[data.ct_series_uid.isna() & data.issues.isna(), "issues"] = "no CT"
    data["global_status"] = np.where(eligible, "ready", "excluded")
    data["region_status"] = np.where(eligible & data.has_seg, "ready", "excluded")
    columns = ["patient_id", "os_time_days", "os_event", "age", "sex", "t_stage", "n_stage", "m_stage", "overall_stage",
               "histology", "ptnm", "affiliation", "ct_series_uid", "ct_study_uid", "ct_series_dir",
               "ct_expected_slices", "ct_found_slices", "ct_max_gap_ratio", "ct_audit_issues", "selection_rule", "description", "slice_thickness_mm", "kernel",
               "manufacturer", "model", "study_date", "csv_ct_date", "ct_date_minus_csv_days", "seg_series_uid",
               "seg_file", "seg_label", "has_seg", "global_status", "region_status", "issues"]
    data[columns].sort_values("patient_id").to_csv(args.output_dir / "cohort.csv", index=False)
    print(data.groupby(["global_status", "region_status"]).size())
    print(data.loc[data.issues.fillna("").ne(""), ["patient_id", "issues"]].to_string(index=False))


if __name__ == "__main__":
    main()
