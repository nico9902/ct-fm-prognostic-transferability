#!/usr/bin/env python3
"""Frozen tumour-centred embeddings for LUNG1 (phase 2.2): Pai et al. 2024 and Hosny et al. 2018.

Seed: physical centroid of our cached primary GTV mask (the same mask used for volume,
radiomics and E4), on the cached 1 mm isotropic HU CT.

* pai:   official FMCIB preprocessing (MONAI: (HU+1024)/3072, 1 mm, LPS, seed-based 50^3
         crop, transpose to z,y,x, zero pad) and MONAI ResNet50 trunk -> 4096-d vector.
* hosny: 150^3 1 mm axis-aligned grid centred on the seed (z,y,x, -1024 HU outside the
         image), central 50^3 crop, HU [-1024, 3071] -> [0, 1]; PyTorch port of the
         HarvardRT CNN -> 256-d dense_2 vector, plus the zero-shot P(dead at 2 years).

Outputs follow the E1 manifest format: outputs/lung1_e1_embeddings/{pai,hosny}/original_ct/.
Checks: our seeds against the Pai repository LUNG1 seeds, and our Hosny probabilities
against the authors' published Maastro (LUNG1) predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import stats

from hosny_deep_prognosis import REPO as HOSNY_REPO, load as load_hosny, normalize as hosny_normalize
from run_lung1_frozen_cox_e1 import COHORT, ROOT, cohort, sha256

CACHE = ROOT / "data/processed/lung1/thorax_hu"
PAI_REPO = ROOT / "external/pai_fmcib/repo"
PAI_WEIGHTS = ROOT / "external/pai_fmcib/weights/model_weights.torch"
OUTPUT = ROOT / "outputs/lung1_e1_embeddings"


def paths(pid: str) -> tuple[Path, Path]:
    return CACHE / pid / f"{pid}_volume.nii.gz", CACHE / pid / f"{pid}_gtv_mask.nii.gz"


def centroid(mask_path: Path) -> np.ndarray:
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(sitk.ReadImage(str(mask_path)) > 0)
    return np.array(stats_filter.GetCentroid(1))


def hosny_patch(image: sitk.Image, seed: np.ndarray) -> np.ndarray:
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((1.0, 1.0, 1.0))
    resampler.SetSize((150, 150, 150))
    resampler.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resampler.SetOutputOrigin(tuple(seed - 74.5))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    array = sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.float32)
    return hosny_normalize(array[50:100, 50:100, 50:100])


def pai_transforms():
    import monai.transforms as T
    spec = importlib.util.spec_from_file_location(
        "fmcib_seed_crop", PAI_REPO / "fmcib/preprocessing/seed_based_crop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return T.Compose([  # fmcib.preprocessing.get_transforms(precropped=False), spatial_size 50^3
        T.LoadImaged(keys=["image_path"], image_only=True, reader="ITKReader"),
        T.EnsureChannelFirstd(keys=["image_path"]),
        T.NormalizeIntensityd(keys=["image_path"], subtrahend=-1024, divisor=3072),
        T.Spacingd(keys=["image_path"], pixdim=1, padding_mode="zeros", mode="linear",
                   align_corners=True, diagonal=True),
        T.Orientationd(keys=["image_path"], axcodes="LPS"),
        module.SeedBasedPatchCropd(keys=["image_path"], roi_size=(50, 50, 50),
                                   coord_orientation="LPS", global_coordinates=True),
        T.SelectItemsd(keys=["image_path"]),
        T.Transposed(keys=["image_path"], indices=(0, 3, 2, 1)),
        T.SpatialPadd(keys=["image_path"], spatial_size=(50, 50, 50)),
    ])


def pai_model(device: torch.device):
    from monai.networks.nets import resnet50
    trunk = resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2,
                     feed_forward=False, bias_downsample=True)
    state = torch.load(PAI_WEIGHTS, map_location="cpu", weights_only=True)["trunk_state_dict"]
    missing = trunk.load_state_dict(state, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise ValueError(f"Pai weights do not match the trunk: {missing}")
    return trunk.to(device).eval()


def write_manifest(name: str, data: pd.DataFrame, vectors: dict[str, np.ndarray], protocol: dict) -> None:
    folder = OUTPUT / name / "original_ct"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, uid in zip(data.patient_id, data.ct_series_uid):
        path = folder / f"{pid}.npy"
        np.save(path, vectors[pid].astype(np.float32))
        rows.append({"patient_id": pid, "ct_series_uid": uid, "embedding_path": str(path)})
    pd.DataFrame(rows).to_csv(folder / "embedding_manifest.csv", index=False)
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    global CACHE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--cache", type=Path, default=CACHE, help="thorax_hu cache root")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--external", action="store_true",
                        help="External cohort: skip the LUNG1-only checks (Pai LUNG1 seeds, Maastro predictions)")
    args = parser.parse_args()
    CACHE, OUTPUT = args.cache, args.output_root
    device = torch.device(args.device)
    data = cohort(args.cohort)
    seeds = {pid: centroid(paths(pid)[1]) for pid in data.patient_id}

    # Seed check against the coordinates shipped with the Pai repository.
    if not args.external:
        published = pd.read_csv(PAI_REPO / "data/preprocessing/lung1/annotations/annotations.csv",
                                dtype={"PatientID": str}).set_index("PatientID")
        distance = pd.Series({pid: float(np.linalg.norm(seeds[pid] - published.loc[pid, ["coordX", "coordY", "coordZ"]]
                                                         .to_numpy(float))) for pid in data.patient_id
                              if pid in published.index})

    hosny = load_hosny().to(device)
    pai = pai_model(device)
    transform = pai_transforms()
    hosny_vectors, pai_vectors, scores = {}, {}, []
    for i, pid in enumerate(data.patient_id):
        volume = paths(pid)[0]
        image = sitk.ReadImage(str(volume))
        x = torch.from_numpy(hosny_patch(image, seeds[pid]))[None].to(device)
        with torch.no_grad():
            logits, features = hosny(x, return_features=True)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        hosny_vectors[pid] = features[0].cpu().numpy()
        seed = seeds[pid]
        sample = transform({"image_path": str(volume), "coordX": seed[0], "coordY": seed[1], "coordZ": seed[2]})
        patch = torch.as_tensor(sample["image_path"]).float()
        if tuple(patch.shape) != (1, 50, 50, 50):
            raise ValueError(f"{pid}: Pai patch shape {tuple(patch.shape)}")
        with torch.no_grad():
            pai_vectors[pid] = pai(patch[None].to(device))[0].cpu().numpy()
        scores.append({"patient_id": pid, "p_dead_2y": float(prob[0]), "p_alive_2y": float(prob[1]),
                       "logit_dead": float(np.log(prob[0]) - np.log(prob[1])),
                       "seed_x": seed[0], "seed_y": seed[1], "seed_z": seed[2]})
        if i % 50 == 0:
            print(f"{i + 1}/{len(data)} {pid}", flush=True)
    for pid in data.patient_id:  # the zero-shot score is also stored as a 1-d "embedding"
        if not (np.isfinite(hosny_vectors[pid]).all() and np.isfinite(pai_vectors[pid]).all()):
            raise ValueError(f"{pid}: non-finite embedding")
    score_table = pd.DataFrame(scores)
    common = {"cohort_sha256": sha256(args.cohort), "seed": "physical centroid of cached primary GTV mask",
              "image": f"cached 1 mm isotropic HU CT ({CACHE.relative_to(ROOT)})"}
    write_manifest("pai", data, pai_vectors, {**common, "model": "Pai et al. 2024 FMCIB trunk",
                                              "weights_sha256": sha256(PAI_WEIGHTS), "dimension": 4096})
    write_manifest("hosny", data, hosny_vectors, {**common, "model": "Hosny et al. 2018 HarvardRT CNN, dense_2",
                                                  "weights_sha256": sha256(HOSNY_REPO / "contrib_src/model/weights.h5"),
                                                  "dimension": 256})
    write_manifest("hosny_score", data, {r["patient_id"]: np.array([r["logit_dead"]]) for r in scores},
                   {**common, "model": "Hosny et al. 2018 zero-shot log-odds of death at 2 years", "dimension": 1})
    score_table.to_csv(OUTPUT / "hosny" / "zero_shot_scores.csv", index=False)
    if args.external:
        return

    # Check against the authors' published LUNG1 predictions (Maastro test cohort).
    maastro = pd.read_csv(HOSNY_REPO / "data/Maastro.csv", usecols=["id", "logit_0"])
    maastro["id"] = maastro.id.str.strip()
    merged = score_table.merge(maastro, left_on="patient_id", right_on="id")
    check = {"n_published_overlap": int(len(merged)),
             "spearman_p_dead_vs_published": float(stats.spearmanr(merged.p_dead_2y, merged.logit_0).statistic),
             "pearson_p_dead_vs_published": float(np.corrcoef(merged.p_dead_2y, merged.logit_0)[0, 1]),
             "median_abs_difference": float(np.median(np.abs(merged.p_dead_2y - merged.logit_0))),
             "rule": "use our preprocessing if Spearman >= 0.9",
             "seed_distance_to_pai_annotations_mm": {"n": int(len(distance)), "median": float(distance.median()),
                                                     "p95": float(distance.quantile(0.95)), "max": float(distance.max())}}
    (OUTPUT / "hosny" / "verification.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
