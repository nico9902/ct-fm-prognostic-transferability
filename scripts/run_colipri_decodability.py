#!/usr/bin/env python3
"""Official frozen COLIPRI-CRM dense features and matched NLST probes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import argparse
import json
import time
import shutil
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/colipri/deps"))
import importlib.metadata
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from run_tangerine_decodability import KEY, TASKS, METRICS, cohort, digest

REVISION = "6f2d04e59c8a2ade62f62f0bd376fd757c966983"
POOLS = ["mean", "max", "attention"]


def native_image(row, directory):
    import torchio as tio
    from convert_nlst_dicom_to_nifti import (read_dicom_slices, filter_localizers,
        filter_slices_by_dominant_shape, sort_slices, deduplicate_identical_positions)
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
    assert image.GetSize()[2] == np.load(row.npy_path, mmap_mode="r").shape[1]
    return tio.ScalarImage.from_sitk(image)


def atomic_save(path, array):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


class PreprocessedVolumes(torch.utils.data.Dataset):
    """CPU workers reproduce the same native-image and official processor path."""
    def __init__(self, d, raw, indices):
        self.d, self.raw, self.indices = d, raw, indices
        self.processor = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        torch.set_num_threads(1)
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        if self.processor is None:
            from colipri import get_processor
            self.processor = get_processor(image_only=True)
        i = self.indices[index]
        row = self.d.iloc[i]
        image = native_image(row, self.raw[tuple(row[k] for k in KEY)])
        return self.processor.process_images(image)[0].data, i


def preprocessed_scans(a, d, raw, processor, cache):
    # Retain the actual shape and official-equivalence smoke test before extraction.
    image = native_image(d.iloc[0], raw[tuple(d.iloc[0][k] for k in KEY)])
    yield 0, image, processor.to_images_batch(processor.process_images(image))
    if a.smoke_only:
        return
    pending = []
    for i in range(1, len(d)):
        paths = [cache / p / f"{i:05d}.npy" for p in POOLS]
        if not all(p.exists() for p in paths) or (not a.no_dense_cache and not (cache / "dense" / f"{i:05d}.npy").exists()):
            pending.append(i)
    loader = DataLoader(PreprocessedVolumes(d, raw, pending), batch_size=1,
        num_workers=a.workers, pin_memory=True,
        **({"multiprocessing_context":"spawn", "prefetch_factor":2} if a.workers else {}))
    for batch, indices in loader:
        yield int(indices[0]), None, batch


def extract(a, d):
    import torchio as tio
    from colipri import get_model, get_processor
    from safetensors import safe_open
    model = get_model(pretrained=False, image_only=True)
    # Accelerate's loader can be permissive; verify every instantiated tensor is present.
    with safe_open(a.checkpoint, framework="pt", device="cpu") as f:
        for k, v in model.state_dict().items():
            if k not in f.keys() or tuple(f.get_slice(k).get_shape()) != tuple(v.shape):
                raise RuntimeError(f"Missing or incompatible checkpoint tensor: {k}")
    from safetensors.torch import load_file
    state = load_file(str(a.checkpoint))
    model.load_state_dict({k:state[k] for k in model.state_dict()}, strict=True)
    del state
    model.requires_grad_(False).eval().to(a.device)
    assert not any(p.requires_grad for p in model.parameters())
    processor = get_processor(image_only=True)
    prep = pd.read_csv(a.prepared_manifest, dtype=str)
    assert not prep.duplicated(KEY).any()
    raw = {tuple(r[k] for k in KEY): r.dicom_dir_resolved for _, r in prep.iterrows()}
    cache = a.output_dir / "cache"
    for pool in POOLS:
        (cache / pool).mkdir(parents=True, exist_ok=True)
    if not a.no_dense_cache:
        (cache / "dense").mkdir(exist_ok=True)
    begin = time.monotonic()
    for i, image, cpu_batch in preprocessed_scans(a, d, raw, processor, cache):
        files = [cache / p / f"{i:05d}.npy" for p in POOLS]
        batch = cpu_batch.to(a.device, non_blocking=True)
        with torch.inference_mode():
            dense = model.encode_image(batch, pool=False, project=False, normalize=False)
            tokens = dense.flatten(2).transpose(1, 2)
            projected = model.image_encoder.projector(dense).flatten(2).transpose(1, 2)
            pooled = {"mean": tokens.mean(1), "max": tokens.amax(1),
                      "attention": model.image_encoder.pooler(projected)}
            if i == 0:
                official = model.encode_image(batch, pool=True, project=True, normalize=False)
                torch.testing.assert_close(pooled["attention"], official)
        if i == 0:
            resampled = tio.Resample(2, image_interpolation="linear", antialias=False)(tio.ToOrientation("RAS")(image))
            smoke = {"original_ct_shape": list(image.shape), "original_spacing_mm": list(image.spacing),
                     "resampled_ct_shape": list(resampled.shape), "colipri_input_shape": list(batch.shape),
                     "dense_grid_shape": list(dense.shape), "dense_token_shape": list(tokens.shape),
                     "pooled_embedding_shape": {p:list(z.shape) for p,z in pooled.items()},
                     "smoke_seconds": time.monotonic()-begin, "official_attention_equivalence": True}
            (a.output_dir / "smoke_shapes.json").write_text(json.dumps(smoke, indent=2))
            print(json.dumps(smoke), flush=True)
        assert torch.isfinite(dense).all()
        for path, pool in zip(files, POOLS):
            z = pooled[pool].float().cpu().numpy()[0]
            assert np.isfinite(z).all()
            atomic_save(path, z)
        if not a.no_dense_cache:
            atomic_save(cache / "dense" / f"{i:05d}.npy", dense.cpu().numpy()[0])
        if a.smoke_only:
            return
        if i % 10 == 0:
            print(f"encoded {i+1}/{len(d)} elapsed={time.monotonic()-begin:.1f}s", flush=True)
    metadata = d[KEY].rename(columns={"timepoint":"screening_round"}).copy()
    metadata["embedding_index"] = np.arange(len(d))
    for pool, dim in [("mean",864),("max",864),("attention",768)]:
        out = a.output_dir / pool / "cache"
        out.mkdir(parents=True, exist_ok=True)
        z = np.stack([np.load(cache / pool / f"{i:05d}.npy") for i in range(len(d))])
        assert z.shape == (len(d),dim) and np.isfinite(z).all()
        atomic_save(out / "embeddings.npy", z)
        metadata.to_csv(out / "embedding_metadata.csv", index=False)
        np.savez_compressed(out / "scan_embeddings.npz", embedding=z,
            **{k:metadata[k].to_numpy(dtype=str) for k in ["patient_id","screening_round","series_uid"]})


def probes(a, d):
    # Match the original linear study, including evaluation at the early-stopping epoch
    # (the original study did not restore the minimum-validation-loss checkpoint).
    from run_diagnostic_state_decodability import DiagnosticLightningModule, metrics, seed_all
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    x = np.load(a.output_dir / "cache/embeddings.npy")
    dim = x.shape[1]
    assert x.shape == (len(d), dim) and np.isfinite(x).all()
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
        for kind in (["linear", "mlp"] if a.mlp else ["linear"]):
            for seed in range(5):
                if any(r["task"] == task and r["model"] == kind and r["seed"] == seed for r in rows):
                    continue
                seed_all(seed)
                module = DiagnosticLightningModule(dim, float((1-y[tr]).sum()/y[tr].sum()))
                if kind == "mlp":
                    module.model = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/ssd2/domenico/datasets/nlst/cohort_1000_setup/staged/diagnostic_decodability"))
    p.add_argument("--prepared-manifest", type=Path, default=Path("/ssd2/domenico/datasets/nlst/processed/nlst/pilot_1000_balanced/sliding_npy_manifest.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/colipri_decodability"))
    p.add_argument("--checkpoint", type=Path, default=Path("external/colipri/model.safetensors"))
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--no-dense-cache", action="store_true")
    p.add_argument("--mlp", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d = cohort(a.root)
    meta = {"cohort_sha256":digest(a.root / "experiment_scan_manifest.csv"),
            "split_sha256":digest(a.root / "patient_splits.csv"),
            "checkpoint_sha256":digest(a.checkpoint), "revision":REVISION,
            "packages":{k:importlib.metadata.version(k) for k in ["colipri","torchio","dynamic-network-architectures","timm","torch"]},
            "encoder_frozen":True, "attention_frozen":True, "source":"original QC-selected DICOM HU",
            "input":"official processor: RAS, linear resample 2mm, clamp [-1000,1000], scale [-1,1], center crop/pad 192 cubed",
            "pools":{"mean":"raw 864-channel dense tokens", "max":"raw 864-channel dense tokens", "attention":"official frozen 12-head pooler after 864->768 projection; mean query"},
            "dense_dtype":"float32", "dense_cached":not a.no_dense_cache,
            "tasks":TASKS,"seeds":list(range(5)),"mlp":a.mlp,
            "pretraining_overlap":"COLIPRI CM/CRM used NLST images for vision-only MAE; decodability, not independent external generalization"}
    mp = a.output_dir / "metadata.json"
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise RuntimeError("Provenance changed; use a new output directory")
    mp.write_text(json.dumps(meta, indent=2))
    d.to_csv(a.output_dir / "experiment_scan_manifest.csv", index=False)
    shutil.copyfile(a.root / "patient_splits.csv", a.output_dir / "patient_splits.csv")
    if not a.skip_extraction:
        extract(a,d)
    if a.smoke_only:
        return
    for pool in POOLS:
        opts = SimpleNamespace(**vars(a))
        opts.output_dir = a.output_dir / pool
        probes(opts,d)
    summary = pd.concat([pd.read_csv(a.output_dir / pool / "summary_results.csv").assign(pooling=pool) for pool in POOLS])
    summary.to_csv(a.output_dir / "summary_results.csv", index=False)
    baseline = pd.read_csv(a.root / "summary_results.csv")
    baseline = baseline[baseline.representation.ne("sliding_gatedattention")]
    tang_root = Path("outputs/tangerine_decodability")
    tang = pd.read_csv(tang_root / "summary_results.csv") if (tang_root / "summary_results.csv").exists() else None
    if tang is not None:
        td = pd.read_csv(tang_root / "experiment_scan_manifest.csv",dtype=str)
        pd.testing.assert_frame_equal(td[KEY+["split"]], d[KEY+["split"]])
        pd.testing.assert_frame_equal(td[TASKS].astype(float).astype(int), d[TASKS].astype(int))
    for metric in METRICS:
        rows = []
        for task in TASKS:
            b = baseline[baseline.task.eq(task)].sort_values("auroc_mean",ascending=False).iloc[0]
            fmt = lambda r: (f"{r[metric+'_mean']:.3f} ± {r[metric+'_std']:.3f}"
                             if metric+'_mean' in r.index else "n/a")
            row = {"Target":task,"Rad-JEPA":fmt(b),"Rad-JEPA representation":b.representation}
            row["TANGERINE"] = fmt(tang[tang.task.eq(task)&tang.model.eq("linear")].iloc[0]) if tang is not None else "unavailable"
            for pool in POOLS:
                row["COLIPRI "+pool] = fmt(summary[summary.task.eq(task)&summary.model.eq("linear")&summary.pooling.eq(pool)].iloc[0])
            rows.append(row)
        pd.DataFrame(rows).to_csv(a.output_dir / f"comparison_{metric}.csv",index=False)
    print(pd.read_csv(a.output_dir / "comparison_auroc.csv").to_string(index=False),flush=True)
    if a.output_dir == Path("outputs/colipri_decodability"):
        from summarize_colipri_decodability import main as summarize
        summarize()


if __name__ == "__main__":
    main()
