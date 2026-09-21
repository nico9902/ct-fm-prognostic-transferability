#!/usr/bin/env python3
"""Frozen official TANGERINE features on the exact Rad-JEPA diagnostic cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


KEY = ["patient_id", "timepoint", "series_uid"]
TASKS = ["isolation_suspicious", "nodule_ge_4mm", "emphysema", "fibrosis_or_scarring"]
METRICS = ["auroc", "auprc", "mcc", "accuracy", "f1", "precision", "recall", "specificity", "tn", "fp", "fn", "tp"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cohort(root):
    d = pd.read_csv(root / "experiment_scan_manifest.csv", dtype=str)
    s = pd.read_csv(root / "patient_splits.csv", dtype=str)
    assert not d.duplicated(KEY).any() and not s.patient_id.duplicated().any()
    assert d.linkage_qc.eq("ok").all()
    merged = d.drop(columns="split").merge(s[["patient_id", "split"]], on="patient_id", validate="many_to_one", sort=False)
    assert merged[KEY].equals(d[KEY]) and merged.split.equals(d.split)
    for task in TASKS:
        merged[task] = pd.to_numeric(merged[task], errors="raise")
        assert merged[task].isin([0, 1]).all()
    assert set(merged.split) == {"train", "val", "test"}
    return merged


class Volumes(Dataset):
    def __init__(self, d, mode, prepared_manifest):
        self.d = d
        self.mode = mode
        self.raw = {}
        if mode == "original_hu":
            p = pd.read_csv(prepared_manifest, dtype=str)
            assert not p.duplicated(KEY).any()
            self.raw = {tuple(r[k] for k in KEY): r.dicom_dir_resolved for _, r in p.iterrows()}

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        row = self.d.iloc[i]
        source = np.load(row.npy_path, mmap_mode="r")
        assert source.ndim == 4 and source.shape[0] == 1 and source.shape[2:] == (256, 256)
        if self.mode == "prepared_approximation":
            assert row.normalization == "window"
            # The original [-1000,400] clipping and foreground crop are irreversible.
            hu = np.asarray(source[0], dtype=np.float32) * 1400 - 1000
            image = sitk.GetImageFromArray(hu)
        else:
            directory = self.raw[tuple(row[k] for k in KEY)]
            from convert_nlst_dicom_to_nifti import (read_dicom_slices, filter_localizers,
                filter_slices_by_dominant_shape, sort_slices, deduplicate_identical_positions)
            # Preserve the prior QC's diagnostic slice selection. A direct GDCM read
            # can include localizers or duplicate positions from the same series.
            slices = read_dicom_slices(Path(directory))
            slices, _ = filter_localizers(slices)
            slices, _ = filter_slices_by_dominant_shape(slices)
            slices, _, _, positions = sort_slices(slices)
            slices, _ = deduplicate_identical_positions(slices, positions)
            slices, _, _, _ = sort_slices(slices)
            assert all(str(s.SeriesInstanceUID) == row.series_uid for s in slices)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([str(s.filename) for s in slices])
            reader.SetOutputPixelType(sitk.sitkFloat32)
            image = reader.Execute()
            if image.GetSize()[2] != source.shape[1]:
                raise RuntimeError(f"Native depth differs from prior QC at {i}: {image.GetSize()} vs {source.shape}")
        resample = sitk.ResampleImageFilter()
        resample.SetSize((256, 256, 256))
        resample.SetOutputSpacing([n * sp / 256 for n, sp in zip(image.GetSize(), image.GetSpacing())])
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetInterpolator(sitk.sitkBSpline)
        volume = sitk.GetArrayFromImage(resample.Execute(image)).astype(np.float32)
        volume = (np.clip(volume, -1200, 800) + 1200) / 2000
        assert np.isfinite(volume).all()
        return torch.from_numpy(volume[None]), i


def build_model(code, checkpoint, device):
    sys.path.insert(0, str(code.resolve()))
    import models_vit
    model = models_vit.vit_large_patch16_yo(img_size=256, num_classes=0, global_pool=True, drop_path_rate=0)
    # Official release is a trusted-source research checkpoint; inspect tensor mapping.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    md = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
    load = model.load_state_dict(filtered, strict=False)
    # Unlike official fine-tuning's permissive filter, never silently leave an encoder random.
    missing = set(load.missing_keys) - {"fc_norm.weight", "fc_norm.bias"}
    if missing:
        raise RuntimeError(f"Missing pretrained encoder weights: {sorted(missing)}")
    model.requires_grad_(False).eval().to(device)
    info = {"checkpoint_sha256": digest(checkpoint), "missing_keys": load.missing_keys,
            "ignored_keys": sorted(set(state) - set(filtered)), "loaded_keys": len(filtered),
            "tensor_shapes": {k: list(v.shape) for k, v in state.items() if torch.is_tensor(v)},
            "trainable_encoder_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, info


def extract(a, d):
    cache = a.output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    model, info = build_model(a.code_dir, a.checkpoint, device)
    info.update({"source_mode": a.source_mode, "cohort_sha256": digest(a.root / "experiment_scan_manifest.csv"),
                 "split_sha256": digest(a.root / "patient_splits.csv"),
                 "official_commit": subprocess.check_output(["git", "-C", str(a.code_dir), "rev-parse", "HEAD"], text=True).strip(),
                 "representation": "pre_logits: fc_norm(CLS + mean patch tokens)", "dim": 2048,
                 "fc_norm_eps": model.fc_norm.eps,
                 "fc_norm_pretrained": not any(k.startswith("fc_norm.") for k in info["missing_keys"]),
                 "torch": torch.__version__, "device": str(device), "dtype": "float32", "masking": False})
    provenance = cache / "encoder_metadata.json"
    if provenance.exists():
        old = json.loads(provenance.read_text())
        for key in ["checkpoint_sha256", "cohort_sha256", "split_sha256", "official_commit", "source_mode", "representation", "dtype"]:
            if old[key] != info[key]:
                raise RuntimeError(f"Cached encoder provenance differs at {key}; use a new output directory")
    ds = Volumes(d, a.source_mode, a.prepared_manifest)
    x, _ = ds[0]
    shapes = {"prepared_input": list(np.load(d.iloc[0].npy_path, mmap_mode="r").shape),
              "preprocessed_input": list(x.shape), "batched_input": [1, *x.shape],
              "patch_grid": [16, 16, 16], "pos_embed": list(model.pos_embed.shape)}
    def hook(name):
        return lambda m, args, output: shapes.update({name: list(output.shape)})
    hs = [model.patch_embed.register_forward_hook(hook("patch_tokens")), model.blocks[-1].register_forward_hook(hook("encoder_tokens"))]
    begin = time.monotonic()
    with torch.inference_mode():
        z = model.fc_norm(model.forward_features(x[None].to(device))).float().cpu().numpy()
    for h in hs:
        h.remove()
    shapes.update({"cls": [1, 1024], "mean_patch": [1, 1024], "whole_ct_embedding": list(z.shape),
                   "linear_logits": [1, 1], "mlp_hidden": [1, 128], "mlp_logits": [1, 1],
                   "smoke_seconds": time.monotonic() - begin, "finite_embedding": bool(np.isfinite(z).all())})
    assert z.shape == (1, 2048) and np.isfinite(z).all()
    (a.output_dir / "smoke_shapes.json").write_text(json.dumps(shapes, indent=2))
    print(json.dumps(shapes), flush=True)
    provenance.write_text(json.dumps(info, indent=2))
    if a.smoke_only:
        return
    # Individual atomic scan files permit interruption/restart without changing the cohort.
    scan_dir = cache / "scans"
    scan_dir.mkdir(exist_ok=True)
    pending = []
    for i, r in d.iterrows():
        p = scan_dir / f"{i:05d}.npy"
        if p.exists():
            e = np.load(p)
            assert e.shape == (2048,) and np.isfinite(e).all()
        else:
            pending.append(i)
    loader = DataLoader(torch.utils.data.Subset(ds, pending), batch_size=a.encode_batch, num_workers=a.workers,
                        pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    completed = len(d) - len(pending)
    with torch.inference_mode():
        for xb, indices in loader:
            emb = model.fc_norm(model.forward_features(xb.to(device, non_blocking=True))).float().cpu().numpy()
            assert emb.shape == (len(indices), 2048) and np.isfinite(emb).all()
            for i, e in zip(indices.tolist(), emb):
                tmp = scan_dir / f"{i:05d}.tmp.npy"
                np.save(tmp, e)
                tmp.replace(scan_dir / f"{i:05d}.npy")
            completed += len(indices)
            print(f"encoded {completed}/{len(d)} elapsed={time.monotonic()-begin:.1f}s", flush=True)
    embeddings = np.stack([np.load(scan_dir / f"{i:05d}.npy") for i in range(len(d))])
    np.save(cache / "embeddings.npy", embeddings)
    m = d[KEY].rename(columns={"timepoint": "screening_round"}).copy()
    m["embedding_index"] = np.arange(len(d))
    m.to_csv(cache / "embedding_metadata.csv", index=False)
    # Self-contained cache requested by the user, with one embedding per keyed scan.
    np.savez_compressed(cache / "scan_embeddings.npz", patient_id=m.patient_id.to_numpy(dtype=str),
                        screening_round=m.screening_round.to_numpy(dtype=str),
                        series_uid=m.series_uid.to_numpy(dtype=str), embedding=embeddings)


def probes(a, d):
    # Match the original linear study, including evaluation at the early-stopping epoch
    # (the original study did not restore the minimum-validation-loss checkpoint).
    from run_diagnostic_state_decodability import DiagnosticLightningModule, metrics, seed_all
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    x = np.load(a.output_dir / "cache/embeddings.npy")
    assert x.shape == (len(d), 2048) and np.isfinite(x).all()
    em = pd.read_csv(a.output_dir / "cache/embedding_metadata.csv", dtype=str)
    assert em.rename(columns={"screening_round": "timepoint"})[KEY].equals(d[KEY])
    tr, va, te = [d.split.eq(s).to_numpy() for s in ["train", "val", "test"]]
    scaler = StandardScaler().fit(x[tr])
    x = scaler.transform(x).astype(np.float32)
    np.savez(a.output_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    rows = []
    path = a.output_dir / "per_seed_results.csv"
    if path.exists():
        rows = pd.read_csv(path).to_dict("records")
    for task in TASKS:
        y = d[task].to_numpy(dtype=np.float32)
        for kind in ["linear", "mlp"]:
            for seed in range(5):
                if any(r["task"] == task and r["model"] == kind and r["seed"] == seed for r in rows):
                    continue
                seed_all(seed)
                module = DiagnosticLightningModule(2048, float((1-y[tr]).sum()/y[tr].sum()))
                if kind == "mlp":
                    module.model = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))
                sets = [TensorDataset(torch.from_numpy(x[ix]), torch.from_numpy(y[ix])) for ix in [tr, va, te]]
                train = DataLoader(sets[0], batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
                val = DataLoader(sets[1], batch_size=128)
                trainer = pl.Trainer(accelerator="gpu" if a.device.startswith("cuda") else "cpu",
                    devices=[int(a.device.split(":")[-1])] if a.device.startswith("cuda") else 1,
                    max_epochs=100, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, deterministic=True,
                    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=15, min_delta=1e-5)])
                trainer.fit(module, train, val)
                module.eval()
                with torch.inference_mode():
                    prob = torch.sigmoid(module(torch.from_numpy(x[te]).to(module.device))).cpu().numpy()
                result = metrics(y[te], prob)
                run = a.output_dir / "runs" / f"{task}_{kind}_{seed}"
                run.mkdir(parents=True, exist_ok=True)
                torch.save(module.state_dict(), run / "probe.pt")
                predictions = d.loc[te, KEY].copy()
                predictions["label"] = y[te]
                predictions["probability"] = prob
                predictions.to_csv(run / "test_predictions.csv", index=False)
                rows.append({"task": task, "model": kind, "seed": seed, "epochs_run": trainer.current_epoch,
                             "n_test": int(te.sum()), **result})
                pd.DataFrame(rows).to_csv(path, index=False)
                print(task, kind, seed, result, flush=True)
    out = pd.DataFrame(rows)
    agg = out.groupby(["task", "model"])[METRICS].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.to_csv(a.output_dir / "summary_results.csv", index=False)
    baseline = pd.read_csv(a.root / "summary_results.csv")
    baseline = baseline[baseline.representation.ne("sliding_gatedattention")]
    comparison = []
    for task in TASKS:
        b = baseline[baseline.task.eq(task)].sort_values("auroc_mean", ascending=False).iloc[0]
        r = {"Target": task, "Rad-JEPA representation": b.representation,
             "Rad-JEPA best linear": f"{b.auroc_mean:.3f} ± {b.auroc_std:.3f}"}
        for kind in ["linear", "mlp"]:
            z = agg[agg.task.eq(task) & agg.model.eq(kind)].iloc[0]
            r[f"TANGERINE {kind}"] = f"{z.auroc_mean:.3f} ± {z.auroc_std:.3f}"
        comparison.append(r)
    pd.DataFrame(comparison).to_csv(a.output_dir / "comparison.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability"))
    p.add_argument("--prepared-manifest", type=Path, default=Path("/ssd2/domenico/datasets/nlst/processed/nlst/pilot_1000_balanced/sliding_npy_manifest.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/tangerine_decodability"))
    p.add_argument("--code-dir", type=Path, default=Path("external/tangerine"))
    p.add_argument("--checkpoint", type=Path, default=Path("external/tangerine/tangerine.pth"))
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--source-mode", choices=["original_hu", "prepared_approximation"], default="original_hu")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--encode-batch", type=int, default=1)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--skip-extraction", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d = cohort(a.root)
    metadata = {"cohort_sha256": digest(a.root / "experiment_scan_manifest.csv"),
                "split_sha256": digest(a.root / "patient_splits.csv"), "source_mode": a.source_mode,
                "tasks": TASKS, "seeds": list(range(5)), "encoder_frozen": True,
                "embedding_type": "pre_logits",
                "scan_counts": d.groupby("split").size().to_dict(),
                "patient_counts": d.groupby("split").patient_id.nunique().to_dict(),
                "pretraining_overlap": "Possible: TANGERINE pretraining included NLST-LSS; patient membership unverified.",
                "evaluation": "Original Rad-JEPA protocol: early-stopping epoch, no best-checkpoint restoration; threshold 0.5; sample std ddof=1",
                "linear": "Linear(2048,1)", "mlp": "Linear(2048,128),ReLU,Dropout(0.2),Linear(128,1)"}
    mp = a.output_dir / "metadata.json"
    if mp.exists() and json.loads(mp.read_text()) != metadata:
        raise RuntimeError("Existing cache/probes belong to a different cohort or preprocessing configuration; use a new output directory")
    mp.write_text(json.dumps(metadata, indent=2))
    d.to_csv(a.output_dir / "experiment_scan_manifest.csv", index=False)
    pd.read_csv(a.root / "patient_splits.csv", dtype=str).to_csv(a.output_dir / "patient_splits.csv", index=False)
    if not a.skip_extraction:
        extract(a, d)
    if not a.smoke_only:
        probes(a, d)


if __name__ == "__main__":
    main()
