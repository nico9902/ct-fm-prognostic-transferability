# CT foundation model prognostic transferability

Research code for **Prognostic transferability of CT foundation models in non-small-cell lung cancer: adaptation, tumour dependence, incremental clinical value and external evaluation**.

[Code guide](scripts/README.md) · [Main results](#main-results)

The study compares COLIPRI, TANGERINE, Rad-JEPA-3D, CT-CLIP and DINOv3 for overall-survival prediction on **LUNG1**, with external evaluation on **NSCLC-Radiogenomics**. Analyses cover frozen representations, survival-specific adaptation, training-set size, tumour masking and incremental value beyond clinical and radiomic predictors. The manuscript is a working draft and is maintained separately from this code repository.

## Main results

Uno's concordance at 36 months. Internal values are mean-fold estimates; external intervals are 95% bootstrap intervals conditional on the frozen models. Primary frozen encoders use PCA16 and a regularized Cox head.

| Model | LUNG1 | External (95% interval) |
| --- | ---: | ---: |
| COLIPRI | 0.617 | 0.627 (0.540–0.719) |
| TANGERINE | 0.599 | 0.649 (0.550–0.741) |
| DINOv3 | 0.575 | 0.643 (0.556–0.730) |
| Rad-JEPA-3D | 0.539 | 0.530 (0.436–0.627) |
| CT-CLIP | 0.530 | 0.482 (0.388–0.581) |
| Stage + tumour volume | 0.588 | 0.605 (0.508–0.702) |
| GTV radiomics | 0.591 | 0.617 (0.508–0.723) |
| Clinical + tumour volume | 0.565 | 0.646 (0.547–0.749) |

- **Frozen representations:** COLIPRI ranked highest internally, but no encoder significantly outperformed stage plus volume or radiomics after Holm correction.
- **Adaptation:** trainable heads and encoder adaptation did not improve internal discrimination over the frozen linear head; encoders showed no advantage in label efficiency.
- **Tumour dependence:** hiding the tumour changed discrimination by at most 0.013, despite a marked change in COLIPRI's representation.
- **External evaluation:** discrimination transported for the stronger encoders, but absolute risks were miscalibrated. Predicted 36-month mortality ranged from 37% to 85%, versus 25% observed.

These results do not establish equivalence where comparisons were inconclusive. Exploratory external LoRA results are separate from the frozen benchmark.

## Repository guide

| Location | Contents |
| --- | --- |
| [`scripts/`](scripts/README.md) | Cohort preparation, model extraction, survival experiments and analysis |
| [`configs/lung1_e1_folds.csv`](configs/lung1_e1_folds.csv) | Locked patient fold assignments |
| [`tests/`](tests/) | Survival metrics, Cox fitting and adaptation component checks |
| [`third_party/ctclip/`](third_party/ctclip/README.md) | Documented CT-CLIP compatibility code |

`data/`, `external/`, `outputs/`, `logs/`, `wandb/` and virtual environments are local, ignored directories. They contain datasets, downloaded model code/weights, derived features and run artifacts, and are not included in a fresh clone. Existing analysis output paths are retained for reproducibility.

## Environment

Run commands from the repository root. The full encoder stack targets Linux with CUDA; Rad-JEPA additionally needs the Mamba CUDA extensions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

For the CUDA 12.1 installation workflow, see [`scripts/install_radjepa_cuda_deps.sh`](scripts/install_radjepa_cuda_deps.sh). Additional environments/dependencies are described in [`requirements-colipri.txt`](requirements-colipri.txt) and [`requirements-radiomics.txt`](requirements-radiomics.txt). COLIPRI and TANGERINE require different `timm` versions; follow the isolated dependency instructions in the COLIPRI requirements.

## Reproduction workflow

1. Obtain the LUNG1 and NSCLC-Radiogenomics data and pretrained weights separately.
2. Prepare the cohorts and model inputs using the entry points in the [script guide](scripts/README.md). Review each script's arguments and input paths before launching it.
3. Extract embeddings and run the locked-fold analyses.

A clone alone cannot reproduce the experiments without the datasets, external encoder repositories and checkpoints. The locations expected by the loaders are defined in `scripts/extract_lung1_e1_embeddings.py`. Manuscript sources, protocols and reports are maintained locally and are not distributed in this repository.

## Checks

```bash
.venv/bin/python -m pytest tests -q
```
