#!/usr/bin/env python3
"""Cache E1-identical original-CT input tensors for image-level E2 adaptation."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_lung1_e1_embeddings import (  # noqa: E402
    COHORT, CHECKPOINTS, ctclip_input, dinov3_input, radjepa_input, read_image,
    sha256, tangerine_input,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--status-column", choices=("region_status", "global_status"), default="region_status")
    parser.add_argument("--clamp-min-hu", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/lung1_e2_inputs")
    args = parser.parse_args()
    if args.max_patients < 0:
        parser.error("max-patients must be nonnegative")
    if args.model == "ctclip" and not torch.cuda.is_available():
        raise RuntimeError("CT-CLIP preprocessing requires a visible CUDA GPU")
    data = pd.read_csv(args.cohort, dtype={"patient_id": str, "ct_series_uid": str})
    data = data.loc[data[args.status_column].eq("ready")].sort_values("patient_id").reset_index(drop=True)
    if args.max_patients:
        data = data.head(args.max_patients)
    output = args.output_root / args.model / "original_ct"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": args.model, "source": "original_ct", "cohort_sha256": sha256(args.cohort),
                "checkpoint_sha256": sha256(CHECKPOINTS[args.model]),
                "input_preprocessing": "extract_lung1_e1_embeddings.py", "dtype": "float16",
                "max_patients": args.max_patients}
    if args.cohort != COHORT or args.status_column != "region_status" or args.clamp_min_hu is not None:
        protocol.update(status_column=args.status_column, clamp_min_hu=args.clamp_min_hu)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing input cache has different provenance")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    processor = None
    if args.model == "colipri":
        from colipri import get_processor
        processor = get_processor(image_only=True)
    records = []
    for index, row in data.iterrows():
        path = output / f"{row.patient_id}.npy"
        if path.exists():
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        else:
            image = read_image(row, "original_ct", ROOT / "data/processed/lung1/thorax_hu")
            if args.clamp_min_hu is not None:
                import SimpleITK as sitk
                image = sitk.Maximum(image, float(args.clamp_min_hu))
            if args.model == "tangerine":
                batch = tangerine_input(image)
            elif args.model == "radjepa":
                batch = radjepa_input(image)
            elif args.model == "dinov3":
                batch = dinov3_input(image)
            elif args.model == "ctclip":
                batch = ctclip_input(image, torch.device(args.device)).cpu()
            else:
                import torchio as tio
                batch = processor.to_images_batch(processor.process_images(tio.ScalarImage.from_sitk(image)))
            array = batch[0, 0].numpy().astype(np.float16)
            temporary = output / f"{row.patient_id}.tmp.npy"
            np.save(temporary, array)
            temporary.replace(path)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError(f"Invalid cached CT tensor: {path}")
        records.append({"patient_id": row.patient_id, "ct_series_uid": row.ct_series_uid,
                        "input_path": path.name, "shape": json.dumps(array.shape)})
        temporary = output / "input_manifest.tmp.csv"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output / "input_manifest.csv")
        print(f"{args.model}: {index + 1}/{len(data)} {row.patient_id}", flush=True)


if __name__ == "__main__":
    main()
