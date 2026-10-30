#!/usr/bin/env python3
"""Extract fixed, outcome-blind PyRadiomics features from cached LUNG1 GTVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/derived/lung1/cohort_with_gtv_volume.csv"
INPUT = ROOT / "data/processed/lung1/thorax_hu"
OUTPUT = ROOT / "outputs/lung1_e3_baselines/radiomics_features.csv"
EXTRACTOR = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(patient_id: str) -> tuple[Path, Path]:
    folder = INPUT / patient_id
    return folder / f"{patient_id}_volume.nii.gz", folder / f"{patient_id}_gtv_mask.nii.gz"


def cropped_images(volume: Path, mask: Path) -> tuple[sitk.Image, sitk.Image]:
    image = sitk.ReadImage(str(volume))
    tumor = sitk.ReadImage(str(mask))
    if image.GetSize() != tumor.GetSize():
        raise ValueError("CT and GTV mask have different grids")
    geometry = sitk.LabelShapeStatisticsImageFilter()
    geometry.Execute(tumor)
    if not geometry.HasLabel(1):
        raise ValueError("GTV mask has no label 1")
    box = geometry.GetBoundingBox(1)
    start = [max(0, int(box[i]) - 5) for i in range(3)]
    stop = [min(image.GetSize()[i], int(box[i] + box[i + 3]) + 5) for i in range(3)]
    size = [b - a for a, b in zip(start, stop)]
    return sitk.RegionOfInterest(image, size, start), sitk.RegionOfInterest(tumor, size, start)


def init_worker() -> None:
    global EXTRACTOR
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(binWidth=25, normalize=False,
                                                             correctMask=False, minimumROIDimensions=2)
    EXTRACTOR.disableAllImageTypes()
    EXTRACTOR.enableImageTypeByName("Original")
    EXTRACTOR.enableAllFeatures()


def extract_one(patient_id: str) -> tuple[str, dict | None, str | None]:
    volume, mask = paths(patient_id)
    if not volume.exists() or not mask.exists():
        return patient_id, None, "missing CT or mask"
    try:
        image, tumor = cropped_images(volume, mask)
        result = EXTRACTOR.execute(image, tumor, label=1)
        record = {"patient_id": patient_id}
        for key, value in result.items():
            if key.startswith("original_"):
                number = float(value)
                record[key] = number if np.isfinite(number) else np.nan
        if not record.keys() - {"patient_id"}:
            raise ValueError("No original-image features returned")
        return patient_id, record, None
    except Exception as exc:
        return patient_id, None, str(exc)


def main() -> None:
    global COHORT, INPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--input", type=Path, default=INPUT, help="thorax_hu cache root")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    COHORT, INPUT = args.cohort, args.input
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_new < 0:
        parser.error("--max-new must be nonnegative")
    data = pd.read_csv(COHORT, dtype={"patient_id": str})
    data = data.loc[data.region_status.eq("ready")].sort_values("patient_id").reset_index(drop=True)
    existing = pd.read_csv(args.output, dtype={"patient_id": str}) if args.output.exists() else pd.DataFrame()
    if not existing.empty and existing.patient_id.duplicated().any():
        raise ValueError("Saved radiomics table has duplicate patient IDs")
    completed = set(existing.patient_id) if not existing.empty else set()
    records = []
    failures = []
    pending = [pid for pid in data.patient_id if pid not in completed]
    if args.max_new:
        pending = pending[:args.max_new]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for i, (patient_id, record, error) in enumerate(pool.map(extract_one, pending), 1):
            if error is not None:
                failures.append((patient_id, error))
            else:
                records.append(record)
            if i % 10 == 0 or i == len(pending):
                print(f"Radiomics {len(completed) + i}/{len(data)}; extracted={len(completed) + i - len(failures)}; failures={len(failures)}", flush=True)
            if len(records) >= 10 or i == len(pending):
                if records:
                    combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
                    if combined.patient_id.duplicated().any():
                        raise ValueError("Duplicate patient feature row")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    combined.to_csv(args.output.with_suffix(".tmp"), index=False)
                    args.output.with_suffix(".tmp").replace(args.output)
                    existing = combined
                    records.clear()
    if failures:
        raise RuntimeError(f"Feature extraction failed: {failures[:10]} ({len(failures)} total)")
    if len(existing) == len(data):
        if set(existing.patient_id) != set(data.patient_id):
            raise ValueError("Radiomics patient IDs do not match E3 cohort")
        feature_names = [name for name in existing if name.startswith("original_")]
        provenance = {"cohort_sha256": sha256(COHORT), "source": str(INPUT),
                      "image": "cached 1 mm isotropic, unmasked thorax CT in HU",
                      "mask": "cached primary-neoplasm GTV", "pyradiomics_image_type": "Original",
                      "bin_width_hu": 25, "n_patients": len(existing),
                      "n_features": len(feature_names), "feature_columns": feature_names}
        args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Complete: {len(existing)} patients, {len(feature_names)} features")


if __name__ == "__main__":
    main()
