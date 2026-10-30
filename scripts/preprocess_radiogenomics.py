#!/usr/bin/env python3
"""Phase 4: thorax_hu cache, GTV mask and GTV volume for the Radiogenomics primary population.

Identical recipe to LUNG1 (preprocess_lung1_luna_style.py --mode thorax_hu: 1 mm isotropic
HU, R231 lungs + GTV bounding box + 10 voxels, unmasked). Only two things differ: voxels
below -1024 HU (out-of-field padding) are set to -1024 at read time, and the SEG reader:
Radiogenomics SEGs hold one binary segment (TCIA labels it Heart/Tissue/Segmentation),
with shared geometry, no SpacingBetweenSlices, and the slice step taken from the frame
positions.

Pre-specified checks (docs/lung1_phase4_protocol.md); a failing patient leaves the
primary population:
* the mask centroid lies within 5 mm of the Pai repository seed;
* the mask overlaps the R231 lung mask dilated by 10 voxels.

GTV volume = native SEG voxels x voxel volume, as add_lung1_gtv_volume.py does for LUNG1.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pydicom.pixels import iter_pixels
from scipy.ndimage import binary_dilation, generate_binary_structure

import preprocess_lung1_luna_style as lung1

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/derived/radiogenomics/cohort.csv"
OUTPUT_COHORT = ROOT / "data/derived/radiogenomics/cohort_with_gtv_volume.csv"
CACHE = ROOT / "data/processed/radiogenomics/thorax_hu"
PAI_SEEDS = ROOT / "external/pai_fmcib/repo/data/preprocessing/radio/annotations/nsclc_radiogenomics.csv"
MAX_SEED_DISTANCE_MM = 5.0


def read_seg(seg_path: Path) -> tuple[sitk.Image, float, int]:
    """Single-segment binary SEG on its own grid, plus native volume (cm3) and voxel count."""
    ds = pydicom.dcmread(seg_path, stop_before_pixels=True)
    if str(ds.SegmentationType).upper() != "BINARY" or len(ds.SegmentSequence) != 1:
        raise ValueError("Expected one binary SEG segment")
    shared = ds.SharedFunctionalGroupsSequence[0]
    orient = np.asarray(shared.PlaneOrientationSequence[0].ImageOrientationPatient, dtype=float)
    row_axis, col_axis = orient[:3], orient[3:]
    normal = np.cross(row_axis, col_axis)
    spacing = [float(x) for x in shared.PixelMeasuresSequence[0].PixelSpacing]
    frames = [(float(np.dot(np.asarray(f.PlanePositionSequence[0].ImagePositionPatient, float), normal)),
               np.asarray(f.PlanePositionSequence[0].ImagePositionPatient, float), i)
              for i, f in enumerate(ds.PerFrameFunctionalGroupsSequence)]
    frames.sort(key=lambda item: item[0])
    depths = np.array([item[0] for item in frames])
    steps = np.diff(depths)
    if len(steps) == 0 or steps.min() <= 0:
        raise ValueError("SEG frames are not distinct planes")
    step = float(np.median(steps))
    index = np.rint((depths - depths[0]) / step).astype(int)
    if np.abs(depths - depths[0] - index * step).max() > 0.1 * step:
        raise ValueError("SEG frame positions are not on a regular grid")
    pixels = {i: np.asarray(p, dtype=np.uint8) for i, p in zip(range(len(frames)), iter_pixels(seg_path))}
    volume = np.zeros((index[-1] + 1, int(ds.Rows), int(ds.Columns)), dtype=np.uint8)
    for k, (_, _, i) in zip(index, frames):
        volume[k] = np.maximum(volume[k], pixels[i])
    voxels = int(volume.sum())
    if voxels == 0:
        raise ValueError("SEG mask is empty")
    mask = sitk.GetImageFromArray(volume)
    mask.SetOrigin(tuple(float(x) for x in frames[0][1]))
    mask.SetSpacing((spacing[1], spacing[0], step))
    mask.SetDirection(tuple(float(x) for x in np.column_stack((row_axis, col_axis, normal)).ravel()))
    return mask, voxels * spacing[0] * spacing[1] * step / 1000.0, voxels


def seg_to_ct_grid(seg_path: Path, ct: sitk.Image) -> sitk.Image:
    mask, _, _ = read_seg(seg_path)
    return sitk.Resample(mask, ct, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


_read_ct = lung1.read_ct


def read_ct(row: dict) -> sitk.Image:
    """LUNG1 reader, with out-of-field padding below -1024 HU set to -1024 (protocol amendment 2)."""
    return sitk.Maximum(_read_ct(row), -1024.0)


lung1.seg_to_ct_grid = seg_to_ct_grid  # the SEG reader and the padding fix are the only changes
lung1.read_ct = read_ct


def checks(pid: str, seeds: pd.DataFrame) -> dict:
    folder = CACHE / pid
    gtv = sitk.ReadImage(str(folder / f"{pid}_gtv_mask.nii.gz")) > 0
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(gtv)
    centroid = np.array(shape.GetCentroid(1))
    lung = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / f"{pid}_lung_mask.nii.gz"))) > 0
    dilated = binary_dilation(lung, structure=generate_binary_structure(3, 1), iterations=10)
    tumour = sitk.GetArrayFromImage(gtv).astype(bool)
    distance = (float(np.linalg.norm(centroid - seeds.loc[pid, ["coordX", "coordY", "coordZ"]].to_numpy(float)))
                if pid in seeds.index else np.nan)
    return {"patient_id": pid, "centroid_x": centroid[0], "centroid_y": centroid[1], "centroid_z": centroid[2],
            "seed_distance_to_pai_mm": distance,
            "gtv_fraction_in_dilated_lung": float((tumour & dilated).sum() / tumour.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--patient-ids", nargs="+")
    args = parser.parse_args()
    data = pd.read_csv(COHORT, dtype={"patient_id": str})
    rows = data.loc[data.region_status.eq("ready")]
    if args.patient_ids:
        rows = rows.loc[rows.patient_id.isin(args.patient_ids)]
    from lungmask import LMInferer
    inferer = LMInferer(modelname="R231", batch_size=args.batch_size, tqdm_disable=True)
    print(f"lungmask device: {inferer.device}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    def run(row: dict) -> dict:
        try:
            result = lung1.process(row, CACHE, inferer, mode="thorax_hu")
            _, volume, voxels = read_seg(Path(row["seg_file"]))
            return {**result, "gtv_volume_cm3": volume, "gtv_mask_voxels": voxels}
        except Exception as exc:  # recorded, the patient leaves the primary population
            print(f"{row['patient_id']}: FAILED: {exc}", flush=True)
            return {"patient_id": row["patient_id"], "error": str(exc)}

    as_text = pd.read_csv(COHORT, dtype=str, keep_default_na=False).set_index("patient_id")
    records = [{"patient_id": pid, **as_text.loc[pid].to_dict()} for pid in rows.patient_id]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, records))
    lung1.write_summary(results, CACHE)
    if args.patient_ids:
        return
    seeds = pd.read_csv(PAI_SEEDS).set_index("Case ID")
    done = [r for r in results if "error" not in r]
    check = pd.DataFrame([checks(r["patient_id"], seeds) for r in done])
    volumes = pd.DataFrame(done)[["patient_id", "gtv_volume_cm3", "gtv_mask_voxels"]]
    errors = {r["patient_id"]: r["error"] for r in results if "error" in r}
    out = data.merge(volumes, on="patient_id", how="left").merge(check, on="patient_id", how="left")
    out["gtv_volume_error"] = out.patient_id.map(errors)
    failed = out.region_status.eq("ready") & (
        out.gtv_volume_error.notna() | ~(out.seed_distance_to_pai_mm <= MAX_SEED_DISTANCE_MM)
        | ~(out.gtv_fraction_in_dilated_lung > 0))
    out.loc[failed, "issues"] = out.loc[failed, "issues"].fillna("").astype(str) + "; failed GTV check"
    out.loc[failed, "region_status"] = "excluded"
    out.to_csv(OUTPUT_COHORT, index=False)
    (CACHE / "checks.json").write_text(json.dumps({
        "n_processed": len(done), "n_failed_processing": len(errors), "n_failed_checks": int(failed.sum()),
        "failed": out.loc[failed, ["patient_id", "seed_distance_to_pai_mm", "gtv_fraction_in_dilated_lung",
                                   "gtv_volume_error"]].to_dict(orient="records"),
        "seed_distance_mm": check.seed_distance_to_pai_mm.describe().to_dict(),
        "gtv_fraction_in_dilated_lung_min": float(check.gtv_fraction_in_dilated_lung.min()),
        "gtv_volume_cm3": volumes.gtv_volume_cm3.describe().to_dict()}, indent=2, default=str) + "\n")
    print(out.region_status.value_counts().to_dict(), "failed checks:", int(failed.sum()))


if __name__ == "__main__":
    main()
