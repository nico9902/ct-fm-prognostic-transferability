# lung-jepa-world-model — NLST data bootstrap

This starter project builds a longitudinal NLST CT cohort for the first
Rad-JEPA-3D temporal/world-model experiments.

## 1. Environment

```bash
cd lung-jepa-world-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the data/download/conversion stack only. The
CUDA/PyTorch stack for Rad-JEPA-3D is installed separately on the server so pip
does not accidentally select an incompatible PyTorch CUDA wheel.

## 2. Test on exactly 100 eligible longitudinal subjects

```bash
python scripts/build_nlst_manifest.py \
  --max-patients 100 \
  --seed 42 \
  --tag test_100
```

`--max-patients` is applied **after** keeping only subjects with at least two
usable CT studies.

Expected outputs:

```text
data/manifests/
├── nlst_test_100_series.csv
├── nlst_test_100_subjects.csv
├── nlst_test_100_pairs.csv
├── nlst_test_100_series_uids.txt
└── nlst_test_100.tcia
```

If the TCIA manifest endpoint is unavailable, the CSVs and UID list are still
created. The UID list can then be used to recreate the download manifest.

To build the cohort manifest, download the selected DICOM series and fetch the
NLST clinical tables in one command:

```bash
python scripts/build_nlst_manifest.py \
  --max-patients 100 \
  --seed 42 \
  --tag test_100 \
  --download \
  --download-clinical
```

Clinical outputs are written under:

```text
data/metadata/nlst/clinical/
├── raw/
├── cohort_subsets/<tag>/
└── clinical_tables_summary_<tag>.csv
```

For the first server-side balanced pilot cohort:

```bash
PYTHON=.venv/bin/python bash scripts/run_server_pilot_500_balanced.sh
```

This selects 500 longitudinally eligible patients balanced by `nlst_canc`
status, downloads their DICOM series and clinical data, converts the scans to
NIfTI and prepares Rad-JEPA-ready `.npy` volumes.

## 3. Rad-JEPA-3D bootstrap

The project is set up to start from the pretrained Rad-JEPA-3D encoder:

```text
external/RadJepa/
external/Rad-Jepa-3D-hf/
├── config.json
└── encoder/encoder.pt
```

Prepare converted NIfTI volumes for the encoder:

```bash
python scripts/prepare_radjepa_volumes.py \
  --nifti-manifest data/processed/nlst/nii_manifest.csv
```

This writes `(1, 32, 256, 256)` float32 arrays in `[0, 1]` to:

```text
data/processed/nlst/radjepa_npy/
data/processed/nlst/radjepa_npy_manifest.csv
```

Extract frozen Rad-JEPA embeddings:

```bash
python scripts/extract_radjepa_embeddings.py --batch-size 1
```

Rad-JEPA-3D requires `mamba_ssm`; on macOS this may need to run on a CUDA/Linux
machine.

On the CUDA server, install the Rad-JEPA runtime with:

```bash
PYTHON=.venv/bin/python bash scripts/install_radjepa_cuda_deps.sh
```

The script intentionally installs CUDA 12.1 PyTorch wheels by default, avoiding
CUDA 13 wheels on servers whose NVIDIA driver only supports CUDA 12.x. It also
installs `requirements-training.txt` after PyTorch, so Lightning and W&B use the
same environment.

Build the latent transition table after embeddings exist:

```bash
python scripts/build_latent_transition_dataset.py \
  --pairs data/manifests/nlst_test_10_pairs.csv \
  --clinical-prsn data/metadata/nlst/clinical/cohort_subsets/test_10/nlst_prsn.csv
```

Train the first latent dynamics baseline:

```bash
python scripts/train_latent_predictor.py \
  --config configs/model/latent_predictor.yaml
```

The training implementation lives under `src/lung_jepa/` and uses PyTorch
Lightning. `scripts/train_latent_predictor.py` is only a thin compatibility
wrapper.

Training now runs a patient-level train/validation/test split by default. In
addition to `loss`, `l1` and `cosine`, it logs persistence baseline metrics
where `z_t` is used directly as the prediction for `z_{t+1}`, plus relative
improvement over that baseline. Each completed experiment writes
`metadata.json`, CSV summaries and `experiment_report.xlsx` into its local
`outputs/` run directory.

Planned follow-up experiments are listed in `docs/next_steps.md`.

## 4. Full eligible cohort

```bash
python scripts/build_nlst_manifest.py --tag full
```

No patient limit is applied.

## 5. Pairing strategy

Default:

```text
T0 -> T1
T1 -> T2
```

Use every possible earlier/later pair with:

```bash
python scripts/build_nlst_manifest.py --pair-mode all
```

which also includes, for example:

```text
T0 -> T2
```

## 6. Initial QC heuristic

For every CT study the script selects one representative volumetric series.

It currently:
- removes descriptions containing LOCALIZER, SCOUT, TOPOGRAM, etc.;
- requires at least 50 images by default;
- prefers large series and descriptions containing AXIAL/CHEST/LUNG;
- de-prioritizes CORONAL/SAGITTAL/MPR reconstructions.

This is intentionally only the **metadata-level first pass**. Before training,
the selected DICOMs should undergo a second QC based on DICOM geometry,
slice spacing/thickness, reconstruction kernel and Rad-JEPA preprocessing.

Change the minimum number of images with:

```bash
--min-images 80
```

## 7. Important methodological point

The first version defines longitudinal eligibility as:

```text
>= 2 distinct usable CT studies for the same PatientID
```

It does not require lung-cancer diagnosis. This retains both relatively stable
trajectories and trajectories containing disease development/progression.

Once NLST clinical/CDAS variables are integrated, cohort definitions can be
expanded to cancer-specific, pre-diagnosis, post-diagnosis, treatment-conditioned
and survival-analysis subsets.
