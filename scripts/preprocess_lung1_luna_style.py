"""Apply the sibling lung_ct_3d2d_synthesis LUNA16 recipe to LUNG1 DICOM CTs.

Recipe copied/adapted from src/prs/luna16_preprocessing.py in
/home/domenico/lung_ct_3d2d_synthesis: 1 mm isotropic CT, [-1200, 600] HU
window to uint8, R231 lung mask, ten-voxel lung dilation, and a ten-voxel
margin around the lung bounding box. LUNG1's DICOM SEG replaces LUNA nodules.

Example:
  /home/domenico/lung_ct_3d2d_synthesis/myenv/bin/python \
    scripts/preprocess_lung1_luna_style.py --patient-ids LUNG1-001 LUNG1-023
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.pixels import iter_pixels
from scipy.ndimage import binary_dilation, generate_binary_structure
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/lung1/luna_style"
THORAX_OUTPUT = ROOT / "data/processed/lung1/thorax_hu"


def lum_trans(hu: np.ndarray) -> np.ndarray:
    """Match the LUNA16 lumTrans: [-1200, 600] HU -> uint8 [0, 255]."""
    scaled = np.clip((hu.astype(np.float32) + 1200.0) / 1800.0, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def resample_image(image: sitk.Image, spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear) -> sitk.Image:
    """Match LUNA16 size/origin/direction policy, with air as the CT fill value."""
    size = [max(1, int(round(n * old / new))) for n, old, new in zip(image.GetSize(), image.GetSpacing(), spacing)]
    filt = sitk.ResampleImageFilter()
    filt.SetOutputSpacing(spacing)
    filt.SetSize(size)
    filt.SetOutputDirection(image.GetDirection())
    filt.SetOutputOrigin(image.GetOrigin())
    filt.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    filt.SetInterpolator(interpolator)
    filt.SetDefaultPixelValue(-1000.0 if interpolator == sitk.sitkLinear else 0.0)
    return filt.Execute(image)


def read_ct(row: dict[str, str]) -> sitk.Image:
    paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(row["ct_series_dir"])
    if len(paths) != int(row["ct_found_slices"]):
        raise ValueError(f"GDCM found {len(paths)} CT slices; audit found {row['ct_found_slices']}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise ValueError("CT did not decode as a 3D volume")
    return image


def seg_to_ct_grid(seg_path: Path, ct: sitk.Image) -> sitk.Image:
    """Build the primary-neoplasm SEG in DICOM LPS coordinates and resample to CT."""
    ds = pydicom.dcmread(seg_path, stop_before_pixels=True)
    segments = [int(s.SegmentNumber) for s in ds.SegmentSequence if "neoplasm" in str(s.SegmentLabel).lower()]
    if len(segments) != 1 or str(ds.SegmentationType).upper() != "BINARY":
        raise ValueError("Expected one binary primary-neoplasm SEG segment")
    fg = ds.SharedFunctionalGroupsSequence[0]
    orient = np.asarray(fg.PlaneOrientationSequence[0].ImageOrientationPatient, dtype=float)
    row_axis, col_axis = orient[:3], orient[3:]
    normal = np.cross(row_axis, col_axis)
    measure = fg.PixelMeasuresSequence[0]
    pixel_spacing = [float(x) for x in measure.PixelSpacing]
    slice_spacing = float(measure.SpacingBetweenSlices)
    frame_indices = [i for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence)
                     if int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == segments[0]]
    if not frame_indices:
        raise ValueError("No primary-neoplasm frames")
    entries = []
    for i, pixels in zip(frame_indices, iter_pixels(seg_path, indices=frame_indices)):
        position = np.asarray(ds.PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient, dtype=float)
        entries.append((float(np.dot(position, normal)), position, np.asarray(pixels, dtype=np.uint8)))
    entries.sort(key=lambda x: x[0])
    steps = np.diff([item[0] for item in entries])
    if len(steps) and (np.min(steps) < 0.95 * slice_spacing or np.max(steps) > 1.05 * slice_spacing):
        raise ValueError("SEG frame positions are not regularly spaced")
    mask = sitk.GetImageFromArray(np.stack([item[2] for item in entries], axis=0))
    mask.SetOrigin(tuple(float(x) for x in entries[0][1]))
    mask.SetSpacing((pixel_spacing[1], pixel_spacing[0], slice_spacing))
    direction = np.column_stack((row_axis, col_axis, normal))
    mask.SetDirection(tuple(float(x) for x in direction.ravel()))
    return sitk.Resample(mask, ct, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def image_from_array(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(reference)
    return out


def bbox_from_mask(mask: np.ndarray, margin=10) -> np.ndarray:
    points = np.where(mask > 0)
    if len(points[0]) == 0:
        raise ValueError("Lung segmentation is empty")
    shape = np.asarray(mask.shape)
    box = np.asarray([[max(0, int(p.min()) - margin), min(int(n), int(p.max()) + margin + 1)]
                      for p, n in zip(points, shape)], dtype=int)
    return box


def crop(image: sitk.Image, box_zyx: np.ndarray) -> sitk.Image:
    start_xyz = [int(box_zyx[2, 0]), int(box_zyx[1, 0]), int(box_zyx[0, 0])]
    size_xyz = [int(box_zyx[2, 1] - box_zyx[2, 0]),
                int(box_zyx[1, 1] - box_zyx[1, 0]),
                int(box_zyx[0, 1] - box_zyx[0, 0])]
    return sitk.RegionOfInterest(image, size_xyz, start_xyz)


def preview(path: Path, full_hu: np.ndarray, processed: np.ndarray, lungs: np.ndarray,
            gtv: np.ndarray, box: np.ndarray, mode: str) -> None:
    tumor_per_slice = gtv.sum(axis=(1, 2))
    full_z = int(np.argmax(tumor_per_slice)) if tumor_per_slice.max() else full_hu.shape[0] // 2
    z = min(max(full_z - int(box[0, 0]), 0), processed.shape[0] - 1)
    full_display = lum_trans(full_hu[full_z])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    axes[0].imshow(full_display, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Full CT, 1 mm, lung window")
    display = lum_trans(processed[z]) if mode == "thorax_hu" else processed[z]
    axes[1].imshow(display, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Thorax crop, unmasked" if mode == "thorax_hu" else "LUNA crop and lung masking")
    axes[2].imshow(display, cmap="gray", vmin=0, vmax=255)
    axes[2].contour(lungs[z], levels=[0.5], colors=["cyan"], linewidths=0.6)
    if np.any(gtv[full_z]):
        tumor_crop = gtv[full_z, box[1, 0]:box[1, 1], box[2, 0]:box[2, 1]]
        axes[2].contour(tumor_crop, levels=[0.5], colors=["red"], linewidths=0.8)
    axes[2].set_title("Cropped CT + lung/GTV contours")
    for ax in axes:
        ax.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def process(row: dict[str, str], output_root: Path, inferer, overwrite=False,
            mode="luna_masked") -> dict[str, object]:
    pid = row["patient_id"]
    directory = output_root / pid
    metadata_path = directory / f"{pid}_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("mode", "luna_masked") != mode:
            raise ValueError(f"Existing output for {pid} uses a different preprocessing mode")
        if not overwrite:
            return previous
    print(f"{pid}: reading DICOM and resampling CT", flush=True)
    ct_native = read_ct(row)
    native_hu = sitk.GetArrayViewFromImage(ct_native)
    if not (-2000 < float(native_hu.min()) < -200 and float(native_hu.max()) > 100):
        raise ValueError("Decoded CT intensities do not look like HU")
    ct = resample_image(ct_native)
    full_hu = sitk.GetArrayFromImage(ct)
    gtv_image = seg_to_ct_grid(Path(row["seg_file"]), ct)
    gtv = sitk.GetArrayFromImage(gtv_image).astype(bool)
    if not gtv.any():
        raise ValueError("GTV is empty after SEG-to-CT alignment")

    print(f"{pid}: inferring R231 lung mask", flush=True)
    lung_labels = inferer.apply(full_hu)
    lung = np.asarray(lung_labels > 0, dtype=np.uint8)
    if mode == "thorax_hu":
        # One 3-D box around both lungs retains the mediastinum. Include the
        # aligned GTV if it extends beyond the lung-derived box.
        box = bbox_from_mask(lung | gtv, margin=10)
    else:
        box = bbox_from_mask(lung, margin=10)
    slices = tuple(slice(int(a), int(b)) for a, b in box)
    if mode == "thorax_hu":
        processed = full_hu[slices]
        retained_after_mask = None
    else:
        expanded = binary_dilation(lung > 0, structure=generate_binary_structure(3, 1), iterations=10)
        processed = (lum_trans(full_hu) * expanded.astype(np.uint8))[slices]
        retained_after_mask = float(np.logical_and(gtv, expanded).sum() / gtv.sum())
    cropped_lung = lung[slices]
    cropped_gtv = gtv[slices].astype(np.uint8)
    retained_in_crop = float(cropped_gtv.sum() / gtv.sum())
    if mode == "thorax_hu" and retained_in_crop != 1.0:
        raise ValueError("GTV extends outside the thorax crop")

    directory.mkdir(parents=True, exist_ok=True)
    reference = crop(ct, box)
    sitk.WriteImage(image_from_array(processed, reference), str(directory / f"{pid}_volume.nii.gz"), True)
    sitk.WriteImage(image_from_array(cropped_lung, reference), str(directory / f"{pid}_lung_mask.nii.gz"), True)
    sitk.WriteImage(image_from_array(cropped_gtv, reference), str(directory / f"{pid}_gtv_mask.nii.gz"), True)
    preview(directory / f"{pid}_preview.png", full_hu, processed, cropped_lung, gtv, box, mode)
    result = {
        "patient_id": pid,
        "source_ct_series_uid": row["ct_series_uid"],
        "mode": mode,
        "recipe": ("1mm isotropic HU, R231 + GTV bounding box + 10 voxels, unmasked"
                   if mode == "thorax_hu" else
                   "LUNA16: 1mm isotropic, HU [-1200,600] -> uint8, R231, 10-voxel dilation, lung bbox + 10 voxels"),
        "source_size_xyz": list(ct_native.GetSize()),
        "source_spacing_xyz_mm": list(ct_native.GetSpacing()),
        "resampled_size_xyz": list(ct.GetSize()),
        "crop_size_xyz": list(reference.GetSize()),
        "crop_spacing_xyz_mm": list(reference.GetSpacing()),
        "crop_origin_lps_mm": list(reference.GetOrigin()),
        "crop_direction_lps": list(reference.GetDirection()),
        "bbox_zyx": box.tolist(),
        "source_hu_min": float(native_hu.min()),
        "source_hu_max": float(native_hu.max()),
        "processed_min": int(processed.min()),
        "processed_max": int(processed.max()),
        "lung_voxels_cropped": int(cropped_lung.sum()),
        "gtv_voxels_resampled": int(gtv.sum()),
        "gtv_fraction_inside_crop": retained_in_crop,
        "gtv_fraction_inside_dilated_lung": retained_after_mask,
        "warning": ("GTV is partly erased in processed CT"
                    if retained_after_mask is not None and retained_after_mask < 0.95 else ""),
    }
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{pid}: wrote {directory}; GTV retained in crop: {retained_in_crop:.1%}", flush=True)
    return result


def write_summary(results: list[dict[str, object]], output_dir: Path) -> None:
    fields = sorted({key for result in results for key in result})
    path = output_dir / "preprocessing_summary.csv"
    temporary = output_dir / "preprocessing_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: str(item["patient_id"])):
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in result.items()})
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("luna_masked", "thorax_hu"), default="luna_masked")
    parser.add_argument("--patient-ids", nargs="+")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel CT loading, lung inference, and output writing")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Lungmask inference slices per batch")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    if args.output_dir is None:
        args.output_dir = THORAX_OUTPUT if args.mode == "thorax_hu" else DEFAULT_OUTPUT
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["region_status"] == "ready"]
    if args.patient_ids:
        wanted = set(args.patient_ids)
        rows = [row for row in rows if row["patient_id"] in wanted]
        missing = wanted - {row["patient_id"] for row in rows}
        if missing:
            raise ValueError(f"Patients absent or not region-ready: {sorted(missing)}")
    if args.max_patients is not None:
        rows = rows[:args.max_patients]
    if not rows:
        raise ValueError("No patients selected")
    from lungmask import LMInferer
    inferer = LMInferer(modelname="R231", force_cpu=args.force_cpu,
                        batch_size=args.batch_size, tqdm_disable=args.workers > 1)
    print(f"lungmask device: {inferer.device}; workers: {args.workers}; batch size: {args.batch_size}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def run_one(row):
        try:
            return process(row, args.output_dir, inferer, overwrite=args.overwrite, mode=args.mode)
        except Exception as exc:
            print(f"{row['patient_id']}: FAILED: {exc}", flush=True)
            return {"patient_id": row["patient_id"], "error": str(exc)}
    if args.workers == 1:
        results = []
        for row in rows:
            results.append(run_one(row))
            if len(results) % 20 == 0:
                write_summary(results, args.output_dir)
                print(f"Completed {len(results)}/{len(rows)} patients", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, row): row["patient_id"] for row in rows}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 20 == 0:
                    write_summary(results, args.output_dir)
                    print(f"Completed {len(results)}/{len(rows)} patients", flush=True)
    write_summary(results, args.output_dir)
    if any("error" in result for result in results):
        raise SystemExit("One or more patients failed; see preprocessing_summary.csv")


if __name__ == "__main__":
    main()
