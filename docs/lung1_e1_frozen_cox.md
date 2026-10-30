# LUNG1 E1: frozen representation prognostic decodability

This implements **Experiment E1 / Property P1** in `docs/planning/research_proposal.md`: frozen global CT embeddings, a low-capacity regularized linear Cox probe, identical patient folds and a censoring-aware evaluation. The encoder is never trained by this code.

## Locked cohort and validation

- `data/derived/lung1/cohort_with_gtv_volume.csv` currently has 416 `region_status=ready` patients and 369 observed deaths. Invalid or excluded patients are omitted before model comparison.
- `configs/lung1_e1_folds.csv` contains five repetitions of five stratified outer folds, locked by patient ID. Event status and early (I/II) versus late (IIIa/IIIb) stage define the strata. The script refuses a changed cohort or fold file.
- Each outer training set has three fixed inner folds. All models use the same assignments. Standardization, 32-component PCA and Cox fitting are repeated inside each training fold only. PCA plus Cox remains linear in the original embedding and caps capacity consistently across encoders. The shared L2 penalizer grid is `0.01, 0.1, 1.0`, selected using inner-fold Uno C.
- The primary metric is Uno's IPCW C-index through 36 months, with the censoring Kaplan–Meier curve estimated from the relevant training fold. Secondary metrics are cumulative/dynamic AUC at 12, 24 and 36 months, IPCW Brier scores integrated from 3 to 36 months, 24-month calibration-in-the-large (mean predicted minus observed survival) and a held-out Cox calibration slope. The scripts reject insufficient IPCW support instead of silently substituting Harrell C.
- Point estimates average the 25 outer-fold scores. Paired patient bootstrap resamples held-out patients within each fixed fold, using identical draws across encoders. Its interval is conditional on the fitted models and fold assignments; it does not describe all model-training uncertainty.

Prepare/verify the split file now:

```bash
.venv/bin/python scripts/run_lung1_frozen_cox_e1.py --prepare-only
```

## Frozen embedding extraction

The local checkout contains TANGERINE, Rad-JEPA-3D, COLIPRI and CT-CLIP checkpoints. `extract_lung1_e1_embeddings.py` follows the model-specific preprocessing implemented in this repository:

| Model | Input after source CT | Global representation |
| --- | --- | --- |
| TANGERINE | B-spline resampling to 256³, HU [-1200, 800] | Official normalized CLS plus mean patch vector, 2048 features |
| Rad-JEPA-3D | RAS, linear resize to 32 × 256 × 256, HU [-1000, 400] | Mean final tokens, 384 features |
| COLIPRI | Official image processor, RAS/2 mm/192³ | Official projected attention pool, 768 features |
| CT-CLIP | Author 0.75 × 0.75 × 1.5 mm trilinear resize, HU [-1000, 1000], central 480 × 480 × 240 crop/pad | Temporal mean, flattened spatial tokens, official projection and L2 norm, 512 features |

`--source original_ct` is the **primary proposal-compatible source**: it uses the selected original DICOM series and each encoder's intended preprocessing. `--source thorax_hu` is a **separate harmonized-crop sensitivity experiment** using the GTV-preserving unmasked NIfTI outputs. That crop is GTV-informed, and results from the two sources must never be mixed as though they were one experiment.

Extract each model on a suitable GPU. The CUDA ordinal may differ from the index shown by `nvidia-smi`:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/extract_lung1_e1_embeddings.py \
  --model tangerine --source original_ct --device cuda:0
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/extract_lung1_e1_embeddings.py \
  --model radjepa --source original_ct --device cuda:0
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/extract_lung1_e1_embeddings.py \
  --model colipri --source original_ct --device cuda:0
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/extract_lung1_e1_embeddings.py \
  --model ctclip --source original_ct --device cuda:0
```

Each extraction saves one `.npy` vector per patient, an incrementally updated `embedding_manifest.csv`, and checkpoint/source provenance. It resumes verified cached vectors and rejects incompatible provenance. For the crop sensitivity analysis, replace `original_ct` with `thorax_hu` and use a separate Cox output directory.

One-patient GPU smoke runs on LUNG1-001 completed for all four encoders with `original_ct`. Rad-JEPA also completed on `thorax_hu`; its prepared tensor exactly matched the repository's established Rad-JEPA NIfTI preprocessing on LUNG1-023. These smoke manifests intentionally contain one patient and are rejected by the full-cohort Cox runner.

The CT-CLIP checkpoint is stored at `external/ctclip/CT-CLIP_v2.pt`, with its SHA-256 and provenance in `external/ctclip/checkpoint_metadata.json`. The [official CT-CLIP repository](https://github.com/ibrahimethemhamamci/CT-CLIP) links the authors' code and checkpoint, but their CT-RATE download requires Hugging Face authentication in this environment. This copy came from a public third-party mirror, so identity with the gated original remains unverified. The image-encoder source is pinned in `third_party/ctclip/README.md`. A one-patient GPU check produced a finite, unit-length 512-feature embedding. The runner requires exactly the locked 416 IDs and CT series UIDs for every included encoder; no missing patients are silently dropped.

## Cox run

```bash
.venv/bin/python scripts/run_lung1_frozen_cox_e1.py \
  --embeddings tangerine=outputs/lung1_e1_embeddings/tangerine/original_ct/embedding_manifest.csv \
  --embeddings radjepa=outputs/lung1_e1_embeddings/radjepa/original_ct/embedding_manifest.csv \
  --embeddings colipri=outputs/lung1_e1_embeddings/colipri/original_ct/embedding_manifest.csv \
  --embeddings ctclip=outputs/lung1_e1_embeddings/ctclip/original_ct/embedding_manifest.csv
```

The output directory contains the protocol with hashes, inner-fold assignments, held-out predictions, fold metrics and a summary with conditional bootstrap intervals and paired differences. The current four-model background run uses `outputs/lung1_e1_frozen_cox_four_models`; `scripts/finish_lung1_e1_background.py` waits for all four embedding manifests before starting Cox fitting. Follow it in `logs/lung1_e1/pipeline.log` and the four model logs in the same folder.

The frozen-Cox runner is CPU-based; GPU is used for encoder inference. `lifelines>=0.30` is required in the analysis environment and is already present in this repository's `.venv`.
