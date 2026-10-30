"""Add GTV volume from TCIA DICOM SEG primary-neoplasm frames to a LUNG1 audit.

Run after build_lung1_cohort.py with: .venv/bin/python scripts/add_lung1_gtv_volume.py
The output is a separate enriched table, leaving the header audit intact.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import pydicom
from pydicom.pixels import iter_pixels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/derived/lung1/cohort_audit.csv"
DEFAULT_OUTPUT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"


def frame_position(frame: pydicom.dataset.Dataset) -> tuple[float, float, float]:
    return tuple(float(x) for x in frame.PlanePositionSequence[0].ImagePositionPatient)


def volume_from_seg(path: Path) -> tuple[float, int]:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    if str(ds.SegmentationType).upper() != "BINARY":
        raise ValueError("SEG is not binary")
    tumor_segments = [int(s.SegmentNumber) for s in ds.SegmentSequence if "neoplasm" in str(s.SegmentLabel).lower()]
    if len(tumor_segments) != 1:
        raise ValueError(f"Expected one primary neoplasm segment, got {tumor_segments}")
    frames = ds.PerFrameFunctionalGroupsSequence
    indices = [i for i, frame in enumerate(frames) if int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == tumor_segments[0]]
    if not indices:
        raise ValueError("Primary neoplasm segment has no frames")
    measure = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
    pixel_area = float(measure.PixelSpacing[0]) * float(measure.PixelSpacing[1])
    spacing = float(measure.SpacingBetweenSlices)
    if spacing <= 0 or pixel_area <= 0:
        raise ValueError("Invalid voxel spacing")
    positions = [frame_position(frames[i]) for i in indices]
    # The image stack may be oblique; compare 3D distances between adjacent frames.
    distances = [sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5 for p, q in zip(positions, positions[1:])]
    if distances and (min(distances) < 0.95 * spacing or max(distances) > 1.05 * spacing):
        raise ValueError(f"Primary segment frame spacing differs from declared spacing: {statistics.median(distances):.3f} vs {spacing:.3f} mm")
    voxels = sum(int(frame.sum()) for frame in iter_pixels(path, indices=indices))
    if voxels <= 0:
        raise ValueError("Primary neoplasm mask is empty")
    return voxels * pixel_area * spacing / 1000.0, voxels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors = []
    for i, row in enumerate(rows, start=1):
        if i == 1 or i % 25 == 0:
            print(f"Computing GTV volume {i}/{len(rows)}", flush=True)
        row["gtv_volume_cm3"] = ""
        row["gtv_mask_voxels"] = ""
        row["gtv_volume_error"] = ""
        if row["region_status"] != "ready":
            continue
        try:
            volume, voxels = volume_from_seg(Path(row["seg_file"]))
            row["gtv_volume_cm3"] = f"{volume:.6f}"
            row["gtv_mask_voxels"] = str(voxels)
        except Exception as exc:
            row["gtv_volume_error"] = str(exc)
            row["region_status"] = "exclude"
            errors.append((row["patient_id"], str(exc)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Computed volumes: {sum(bool(row['gtv_volume_cm3']) for row in rows)}")
    print(f"Errors: {errors}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
