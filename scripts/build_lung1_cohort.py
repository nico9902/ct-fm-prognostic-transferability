"""Audit the TCIA LUNG1 clinical table against the locally downloaded DICOM set.

Run with: .venv/bin/python scripts/build_lung1_cohort.py
This checks DICOM headers only; it does not decode CT pixels or convert GTV masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLINICAL = ROOT / "data/clinical/lung1/NSCLC_Radiomics_Lung1.clinical-version3-Oct_2019.csv"
DEFAULT_METADATA = ROOT / "data/LUNG1/dicom/NSCLC-Radiomics-Version-4-Oct-2020-NBIA-manifest/metadata.csv"
DEFAULT_OUTPUT = ROOT / "data/derived/lung1/cohort_audit.csv"
KNOWN_SLICE_GAPS = {"LUNG1-014", "LUNG1-021", "LUNG1-085"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def files_for_series(metadata_root: Path, series: dict[str, str] | None) -> tuple[Path | None, list[Path]]:
    if series is None:
        return None, []
    directory = metadata_root / series["File Location"]
    return directory, sorted(directory.glob("*.dcm")) if directory.is_dir() else []


def outcome(value: str, *, event: bool = False) -> float | int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if event:
        return int(parsed) if parsed in (0, 1) else None
    return parsed if parsed > 0 else None


def scan_ct(files: list[Path], patient_id: str, series_uid: str) -> tuple[list[str], float | None]:
    issues: list[str] = []
    positions: list[float] = []
    orientations: set[tuple[float, ...]] = set()
    dimensions: set[tuple[int, int]] = set()
    spacings: set[tuple[float, ...]] = set()
    uids: set[str] = set()
    for path in files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(ds.get("PatientID", "")) != patient_id or str(ds.get("SeriesInstanceUID", "")) != series_uid:
                issues.append("ct_header_id_mismatch")
            uids.add(str(ds.get("SOPInstanceUID", "")))
            dimensions.add((int(ds.Rows), int(ds.Columns)))
            spacings.add(tuple(round(float(v), 5) for v in ds.PixelSpacing))
            orientation = tuple(round(float(v), 5) for v in ds.ImageOrientationPatient)
            orientations.add(orientation)
            # Project ImagePositionPatient onto the slice normal, including oblique scans.
            row, col = orientation[:3], orientation[3:]
            normal = (
                row[1] * col[2] - row[2] * col[1],
                row[2] * col[0] - row[0] * col[2],
                row[0] * col[1] - row[1] * col[0],
            )
            positions.append(sum(float(a) * b for a, b in zip(ds.ImagePositionPatient, normal)))
        except Exception:
            issues.append("ct_header_read_error")
    if len(uids) != len(files):
        issues.append("duplicate_ct_sop_uid")
    if len(dimensions) > 1 or len(spacings) > 1 or len(orientations) > 1:
        issues.append("inconsistent_ct_geometry")
    gap_ratio = None
    if len(positions) > 2 and len(orientations) == 1:
        steps = [b - a for a, b in zip(sorted(positions), sorted(positions)[1:])]
        positive = [step for step in steps if step > 0.01]
        if len(positive) != len(steps):
            issues.append("duplicate_slice_position")
        if positive:
            median = statistics.median(positive)
            gap_ratio = max(positive) / median
            if gap_ratio > 1.5:
                issues.append("ct_slice_gap")
    return sorted(set(issues)), gap_ratio


def scan_rtstruct(files: list[Path], patient_id: str, study_uid: str) -> tuple[bool, list[str]]:
    if len(files) != 1:
        return False, ["rtstruct_file_count"]
    try:
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        issues = []
        if str(ds.get("PatientID", "")) != patient_id or str(ds.get("StudyInstanceUID", "")) != study_uid:
            issues.append("rtstruct_header_id_mismatch")
        has_gtv = any(str(roi.ROIName).strip().upper() == "GTV-1" for roi in ds.StructureSetROISequence)
        if not has_gtv:
            issues.append("missing_gtv1_roi")
        return has_gtv, issues
    except Exception:
        return False, ["rtstruct_header_read_error"]


def scan_seg(files: list[Path], patient_id: str, study_uid: str, ct_uid: str) -> list[str]:
    if not files:
        return ["missing_seg"]
    if len(files) != 1:
        return ["seg_file_count"]
    try:
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        issues = []
        if str(ds.get("PatientID", "")) != patient_id or str(ds.get("StudyInstanceUID", "")) != study_uid:
            issues.append("seg_header_id_mismatch")
        if not any(str(ref.SeriesInstanceUID) == ct_uid for ref in ds.get("ReferencedSeriesSequence", [])):
            issues.append("seg_ct_reference_mismatch")
        if not any("neoplasm" in str(seg.SegmentLabel).lower() for seg in ds.SegmentSequence):
            issues.append("seg_missing_primary_tumor_label")
        return issues
    except Exception:
        return ["seg_header_read_error"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clinical = read_csv(args.clinical)
    metadata = read_csv(args.metadata)
    clinical_ids = [row["PatientID"].strip() for row in clinical]
    if len(clinical_ids) != len(set(clinical_ids)):
        raise ValueError("Duplicate patient IDs in clinical table")
    series_by_patient: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for series in metadata:
        series_by_patient[series["Subject ID"]][series["Modality"]].append(series)
    unknown_imaging_ids = set(series_by_patient) - set(clinical_ids)
    if unknown_imaging_ids:
        raise ValueError(f"Imaging IDs missing from clinical data: {sorted(unknown_imaging_ids)}")

    output_rows = []
    for index, clinical_row in enumerate(clinical, start=1):
        if index == 1 or index % 25 == 0:
            print(f"Auditing patient {index}/{len(clinical)}", flush=True)
        patient_id = clinical_row["PatientID"].strip()
        modalities = series_by_patient.get(patient_id, {})
        ct_list, rt_list, seg_list = (modalities.get(modality, []) for modality in ("CT", "RTSTRUCT", "SEG"))
        ct, rt, seg = (items[0] if len(items) == 1 else None for items in (ct_list, rt_list, seg_list))
        ct_dir, ct_files = files_for_series(args.metadata.parent, ct)
        rt_dir, rt_files = files_for_series(args.metadata.parent, rt)
        seg_dir, seg_files = files_for_series(args.metadata.parent, seg)
        issues = []
        if len(ct_list) != 1:
            issues.append("ct_series_count")
        if len(rt_list) != 1:
            issues.append("rtstruct_series_count")
        if ct and len(ct_files) != int(ct["Number of Images"]):
            issues.append("ct_file_count_mismatch")
        if not ct_files:
            issues.append("missing_ct_files")
        if rt and len(rt_files) != int(rt["Number of Images"]):
            issues.append("rtstruct_file_count_mismatch")
        if seg and len(seg_files) != int(seg["Number of Images"]):
            issues.append("seg_file_count_mismatch")
        survival_time = outcome(clinical_row["Survival.time"])
        event = outcome(clinical_row["deadstatus.event"], event=True)
        if survival_time is None:
            issues.append("invalid_os_time")
        if event is None:
            issues.append("invalid_os_event")
        ct_issues, gap_ratio = scan_ct(ct_files, patient_id, ct["Series UID"] if ct else "")
        issues += ct_issues
        has_gtv, rt_issues = scan_rtstruct(rt_files, patient_id, rt["Study UID"] if rt else "")
        issues += rt_issues
        if seg:
            issues += scan_seg(seg_files, patient_id, seg["Study UID"], ct["Series UID"] if ct else "")
        else:
            issues.append("missing_seg")
        if ct and rt and ct["Study UID"] != rt["Study UID"]:
            issues.append("rtstruct_study_mismatch")
        if ct and seg and ct["Study UID"] != seg["Study UID"]:
            issues.append("seg_study_mismatch")

        # Known incomplete CT volumes need adjudication for a whole-chest benchmark.
        if patient_id in KNOWN_SLICE_GAPS:
            issues.append("tcia_reported_missing_slices")
        issues = sorted(set(issues))
        # Conservative primary cohort: a missing/duplicate slice is a CT exclusion.
        # SEG is supplemental; the GTV-1 RTSTRUCT is the required regional annotation.
        fatal = {"ct_series_count", "missing_ct_files", "ct_file_count_mismatch", "invalid_os_time", "invalid_os_event", "ct_header_id_mismatch", "ct_header_read_error", "ct_slice_gap", "tcia_reported_missing_slices", "duplicate_slice_position", "inconsistent_ct_geometry", "duplicate_ct_sop_uid"}
        regional_fatal = {"rtstruct_series_count", "rtstruct_file_count", "rtstruct_file_count_mismatch", "rtstruct_header_id_mismatch", "rtstruct_header_read_error", "missing_gtv1_roi", "rtstruct_study_mismatch"}
        global_status = "exclude" if fatal.intersection(issues) else "ready"
        region_status = "exclude" if global_status == "exclude" or regional_fatal.intersection(issues) else global_status

        output_rows.append({
            "patient_id": patient_id,
            "os_time_days": survival_time if survival_time is not None else "",
            "os_event": event if event is not None else "",
            "age": clinical_row["age"],
            "sex": clinical_row["gender"],
            "t_stage": clinical_row["clinical.T.Stage"],
            "n_stage": clinical_row["Clinical.N.Stage"],
            "m_stage": clinical_row["Clinical.M.Stage"],
            "overall_stage": clinical_row["Overall.Stage"],
            "histology": clinical_row["Histology"],
            "ct_series_uid": ct["Series UID"] if ct else "",
            "ct_study_uid": ct["Study UID"] if ct else "",
            "ct_series_dir": str(ct_dir.resolve()) if ct_dir else "",
            "ct_expected_slices": ct["Number of Images"] if ct else "",
            "ct_found_slices": len(ct_files),
            "ct_max_gap_ratio": round(gap_ratio, 3) if gap_ratio is not None else "",
            "rtstruct_series_uid": rt["Series UID"] if rt else "",
            "rtstruct_file": str(rt_files[0].resolve()) if len(rt_files) == 1 else "",
            "has_gtv1_roi": int(has_gtv),
            "seg_series_uid": seg["Series UID"] if seg else "",
            "seg_file": str(seg_files[0].resolve()) if len(seg_files) == 1 else "",
            "global_status": global_status,
            "region_status": region_status,
            "issues": ";".join(issues),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Clinical SHA256: {hashlib.sha256(args.clinical.read_bytes()).hexdigest()}")
    print(f"Patients: {len(output_rows)}; events: {sum(row['os_event'] == 1 for row in output_rows)}")
    print(f"Global status: {dict(Counter(row['global_status'] for row in output_rows))}")
    print(f"Regional status: {dict(Counter(row['region_status'] for row in output_rows))}")
    print(f"Issues: {dict(Counter(issue for row in output_rows for issue in row['issues'].split(';') if issue))}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
