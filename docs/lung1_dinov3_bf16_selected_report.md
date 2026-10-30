# LUNG1 E1/E2: DINOv3 on the selected 5-fold split

**Run date:** 16 October 2026  
**Protocol:** [one five-fold CV with validation selection](lung1_bf16_selected_protocol.md)

## Setup

DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, natural-image LVD-1689M pretraining) was added as a fifth encoder. The cohort, split and selection rule are unchanged: 416 LUNG1 patients with 369 deaths, repetition zero of `configs/lung1_e1_folds.csv` as the five outer test folds, and inner fold zero of each outer training set as the validation set. The checkpoint and preprocessing were fixed before outcome analysis.

DINOv3 is a 2-D encoder and is applied to axial slices of the same selected original CT series:

- Reorient to LPS, so array slices are in radiological display (anterior up, patient right on image left).
- Linearly resample along z to 3 mm; keep the full in-plane field of view.
- Clip HU to [−1000, 400], map to [0, 1], replicate the gray value over three channels.
- Resize each slice like the official processor (`preprocessor_config.json`): bilinear with antialiasing to 224 × 224, then ImageNet mean/std. On a test slice this matched `DINOv3ViTImageProcessorFast` exactly.

Representations, per rung:

| Rung | DINOv3 representation | Settings (identical to the other FMs) |
| --- | --- | --- |
| Linear Cox | Mean over slices of the final-norm CLS token (768) | 32 PCA components, L2 ∈ {0.01, 0.1, 1.0} |
| MLP Cox | Same 768-d vector | Width 32, dropout 0.1, AdamW 3e-4, 100 epochs |
| SQA | Patch tokens as a slices × 14 × 14 grid, adaptive average pooled to 8 × 8 × 8; CLS and 4 registers excluded | 128-d adapter, 4 queries, 40 epochs |
| LoRA | Encoder on all slices, slice-mean CLS, LayerNorm + linear Cox head | Rank 8, alpha 16 on the 6 linear layers of the last block (112,897 trainable parameters), lr 1e-4, 10 epochs |
| Partial FT | Same forward | Last block trainable (7.09 M trainable parameters with the head), lr 1e-5, 10 epochs |

All neural rungs restore the epoch with maximal validation Uno C at 36 months before the single test evaluation. Training ran with BF16 mixed precision on the A100. LoRA and partial fine-tuning each took about 11 minutes per fold. transformers ≥ 4.56 is needed for DINOv3, so the encoder code runs from `.venv-dinov3` (transformers 4.57.1 layered on `.venv`). Queue: `scripts/launch_lung1_dinov3_bf16_selected.sh`.

## Results

Pooled Uno C at 36 months over the 416 out-of-fold predictions, ranked by −log S(36 months). Intervals: 500 paired patient bootstrap resamples, the same draws and seed as the [baseline report](lung1_bf16_selected_baselines_report.md). They are conditional on the fitted fold models and have no multiplicity adjustment.

| Rung | Pooled Uno C (95% CI) | Mean fold Uno C | AUC 1 / 2 / 3 y | IBS 3–36 m | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| Linear | 0.546 (0.509–0.576) | 0.556 | 0.562 / 0.557 / 0.572 | 0.221 | penalty 1.0, 0.01, 0.01, 0.01, 1.0 |
| MLP | 0.561 (0.529–0.593) | 0.558 | 0.563 / 0.580 / 0.605 | 0.240 | epochs 99, 84, 73, 76, 95 |
| SQA | 0.533 (0.499–0.566) | 0.544 | 0.546 / 0.533 / 0.548 | 0.269 | epochs 38, 19, 14, 39, 12 |
| LoRA | 0.489 (0.452–0.524) | 0.496 | 0.482 / 0.462 / 0.470 | 0.220 | epochs 5, 10, 8, 8, 2 |
| Partial FT | 0.506 (0.471–0.540) | 0.502 | 0.522 / 0.481 / 0.508 | 0.218 | epochs 10, 10, 8, 4, 10 |

Pooled Uno C of all encoders on the same split:

| Rung | TANGERINE | Rad-JEPA-3D | COLIPRI | CT-CLIP | DINOv3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 0.595 | 0.528 | **0.619** | 0.513 | 0.546 |
| MLP | 0.611 | 0.530 | **0.616** | 0.485 | 0.561 |
| SQA | **0.608** | 0.499 | 0.605 | 0.501 | 0.533 |
| LoRA | 0.520 | 0.492 | **0.535** | 0.517 | 0.489 |
| Partial FT | 0.509 | 0.499 | 0.490 | **0.514** | 0.506 |

Baselines: stage + tumor volume 0.557 (0.527–0.591); GTV radiomics 0.570 (0.538–0.602).

Paired differences, DINOv3 minus comparator (same rung):

| Rung | − TANGERINE | − Rad-JEPA-3D | − COLIPRI | − CT-CLIP | − Stage + volume | − Radiomics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | **−0.050 [−0.085, −0.013]** | +0.017 [−0.030, +0.059] | **−0.073 [−0.116, −0.032]** | +0.033 [−0.013, +0.073] | −0.011 [−0.063, +0.034] | −0.024 [−0.072, +0.020] |
| MLP | **−0.050 [−0.087, −0.016]** | +0.030 [−0.010, +0.074] | **−0.055 [−0.098, −0.014]** | **+0.076 [+0.030, +0.118]** | +0.004 [−0.040, +0.047] | −0.009 [−0.051, +0.037] |
| SQA | **−0.075 [−0.115, −0.038]** | +0.034 [−0.012, +0.080] | **−0.072 [−0.114, −0.026]** | +0.032 [−0.020, +0.081] | −0.024 [−0.067, +0.017] | −0.037 [−0.082, +0.008] |
| LoRA | −0.031 [−0.077, +0.008] | −0.003 [−0.054, +0.047] | **−0.046 [−0.093, −0.000]** | −0.029 [−0.081, +0.022] | **−0.068 [−0.113, −0.020]** | **−0.081 [−0.128, −0.033]** |
| Partial FT | −0.004 [−0.050, +0.040] | +0.006 [−0.043, +0.055] | +0.015 [−0.033, +0.063] | −0.009 [−0.062, +0.033] | **−0.051 [−0.098, −0.011]** | **−0.064 [−0.108, −0.020]** |

Within DINOv3, versus its Linear rung: MLP +0.015 [−0.016, +0.048]; SQA −0.013 [−0.049, +0.025]; LoRA −0.057 [−0.103, −0.015]; partial −0.040 [−0.083, +0.003].

## Interpretation

- **Frozen rungs.** DINOv3 sits between the two strongest 3-D CT encoders (TANGERINE, COLIPRI) and the two weakest (Rad-JEPA-3D, CT-CLIP). It is clearly below TANGERINE and COLIPRI in every frozen rung. It is not distinguishable from Rad-JEPA-3D, and from CT-CLIP except in the MLP rung, where it is higher. Its best rung, MLP at 0.561, is at the level of the clinical and radiomics baselines. It does not exceed them.
- **Fold instability.** Linear fold scores range from 0.461 to 0.619, and the selected penalty alternates between 0.01 and 1.0.
- **Fine-tuning.** As for the other encoders, LoRA and partial fine-tuning on about 221 training patients do not help. They fall to chance level and below both baselines. Several folds select the last epoch, so these rungs remain undertrained or unstable under the fixed 10-epoch budget. This is not evidence for a different budget, which would have to be fixed in advance for all encoders.
- **Scope.** The representation is a slice-mean of a natural-image 2-D model. It does not model 3-D context, and the results do not test other aggregations or multi-window inputs.

## Artifacts

- Embeddings, tokens, inputs: `outputs/lung1_e1_embeddings/dinov3/original_ct`, `outputs/lung1_e2_tokens/dinov3/original_ct`, `outputs/lung1_e2_inputs/dinov3/original_ct`.
- Rungs: `outputs/lung1_bf16_selected/<method>/dinov3`, logs in `logs/lung1_bf16_selected/*_dinov3.log`.
- Paired comparisons: `outputs/lung1_bf16_selected/dinov3_paired_uno_c.json` (`scripts/compare_lung1_dinov3.py`).
- W&B project `lung1-prognostic-transferability`: groups `lung1-5cv-bf16-{linear,mlp,sqa}-dinov3` and `lung1-e2-bf16-{lora,partial}-dinov3` (5 fold runs each), and run `5CV-BF16-dinov3-paired-comparison`.
