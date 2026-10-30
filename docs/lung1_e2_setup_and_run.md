# LUNG1 Experiment E2: setup and execution record

**Current protocol (5 October 2026):** The new one-repetition, five-fold
BF16 selected-checkpoint run is specified in
[`lung1_bf16_selected_protocol.md`](lung1_bf16_selected_protocol.md). The
earlier runs documented below are pilot results under superseded selection
rules and must not be pooled with the new run.

## Locked question and cohort

E2 in `docs/planning/research_proposal.md` compares six adaptation rungs for each of TANGERINE, Rad-JEPA-3D, COLIPRI, and CT-CLIP. Its primary contrast is frozen global Linear Cox versus frozen-token Survival Query Adapter (SQA). The MLP, LoRA and partial fine-tuning rungs are secondary; full fine-tuning is optional.

The existing E1 cohort has 416 LUNG1 patients, 369 observed deaths, and a locked five-repetition by five-fold outer CV file at `configs/lung1_e1_folds.csv`. Three inner folds are derived with the E1 script. The same selected original CT series and model-specific preprocessing are used. No GTV-informed crop is mixed into this comparison.

## Step 1: cached global rungs

Four complete global embedding manifests already existed under `outputs/lung1_e1_embeddings/<model>/original_ct`. The E1 Linear Cox run at `outputs/lung1_e1_frozen_cox_four_models` supplies the frozen linear rung without refitting. `scripts/log_lung1_e2_linear_baseline.py` registered its 25 fold metrics and summary for each model in W&B, with the E1 protocol as provenance.

`scripts/run_lung1_e2_mlp.py` fits a shared one-hidden-layer MLP Cox head on those frozen vectors. Standardization is fitted only on the current training split. Two predefined width/dropout/weight-decay configurations are selected by mean inner-fold Uno C. Efron tied-event partial likelihood trains the head; a training-only Breslow baseline gives predicted survival curves. All 25 outer folds have completed for all four models. Each fold records Uno C at 36 months, cumulative/dynamic AUC at one/two/three years, integrated Brier score from three to 36 months, 2-year calibration offset and slope, trainable parameter count, time, selected configuration, and out-of-fold patient predictions.

Run or resume the global queue:

```bash
WANDB_MODE=online bash scripts/launch_lung1_e2_cached.sh
```

The initial runs were written in W&B offline format because the restricted runtime could not see the host credential. All nine E2 Linear, MLP and paired-comparison runs were then synced successfully to `domenico-paolo1999/lung1-prognostic-transferability` using `scripts/sync_lung1_e2_wandb.sh` after verifying the existing `~/.netrc` credential. No key was copied into the repository.

## Step 2: paired assessment of completed rungs

`scripts/compare_lung1_e2.py` validates patient and fold alignment, computes fold-level Uno C using the locked training censoring distribution, and uses 500 paired patient-cluster bootstrap resamples shared across five repetitions. It saves `outputs/lung1_e2_comparison/paired_uno_c.json` and logs condition scores and contrasts to W&B. These intervals are conditional on fitted models and folds and have no multiplicity adjustment.

| Encoder | Linear Uno C | MLP Uno C | MLP minus Linear, paired 95% interval |
| --- | ---: | ---: | ---: |
| TANGERINE | 0.613 | 0.622 | +0.009 [−0.015, +0.030] |
| Rad-JEPA-3D | 0.550 | 0.523 | −0.026 [−0.048, −0.010] |
| COLIPRI | 0.634 | 0.614 | −0.021 [−0.039, −0.003] |
| CT-CLIP | 0.529 | 0.541 | +0.012 [−0.010, +0.033] |

The MLP is a secondary rung. These numbers do **not** answer the primary Linear-to-SQA comparison.

## Step 3: final token cache and SQA

`scripts/extract_lung1_e2_tokens.py` loads the same checkpoint and original CT input path as E1 and caches final spatial tokens without labels. Grids with an axis larger than eight are deterministically average pooled to at most eight per axis. Arrays are float16 on disk, and each manifest records patient/series identity, saved shape, native and saved grid. TANGERINE's class token is excluded; Rad-JEPA's Morton scan is reversed when present; COLIPRI uses its projected dense tokens; CT-CLIP uses its encoded image tokens. One-patient A100 smoke extraction passed for all four models. COLIPRI requires `torch.no_grad()` for its dynamically constructed dense pooling layer.

`scripts/run_lung1_e2_sqa.py` uses a shared 128-dimensional projection, four learned queries, one four-head cross-attention layer, fixed 3D Fourier positions, and the same Efron Cox loss and survival metrics as the MLP rung. Projection width, query count and attention depth are fixed across FMs; only the input projection size changes. It selects between the same two dropout/weight-decay choices on the locked inner folds. Fitting uses full outer-training-set risk sets, with microbatches for forward memory. The token channel normalization is fitted only on the current training split. Each fold logs trainable parameters, peak GPU memory, elapsed time, held-out metrics and predictions to W&B.

The full extraction and SQA nested-CV queue was started in a persistent background session on the A100 with `CUDA_VISIBLE_DEVICES=3` and `WANDB_MODE=online`:

```bash
WANDB_MODE=online CUDA_VISIBLE_DEVICES=3 bash scripts/launch_lung1_e2_sqa_gpu.sh
```

This queue is resumable per patient for extraction and per outer fold for SQA. Watch `logs/lung1_e2/tokens_<model>.log` and `logs/lung1_e2/sqa_<model>.log`. `outputs/lung1_e2_sqa/COMPLETE` appears only after all four SQA runs and the refreshed paired comparison finish. Current extraction and SQA results must not be treated as complete until that marker exists.

## Step 4: LoRA and partial encoder fine-tuning

LoRA and partial encoder fine-tuning require image-level re-encoding for every training update. The cached final tokens cannot validly stand in for those rungs because frozen early layers would prevent encoder changes from reaching the output. `scripts/cache_lung1_e2_inputs.py` therefore caches the E1-identical original-CT input tensor for each patient and model, with patient/series and preprocessing provenance. All four one-patient input checks passed; the full cache is building in a background session on a V100 via `scripts/launch_lung1_e2_inputs_gpu.sh`.

`scripts/run_lung1_e2_encoder_adaptation.py` provides fixed, predeclared settings for both rungs on the locked 25 outer folds. LoRA has rank 8, alpha 16, learning rate 1e-4 and weight decay 1e-4. Partial fine-tuning unfreezes the final selected block, uses learning rate 1e-5 and weight decay 1e-4. Both use 10 epochs, an Efron Cox partial likelihood and the same survival metrics. The selected block is the last ViT block for TANGERINE, last Mamba block for Rad-JEPA-3D, and last EVA block for COLIPRI. CT-CLIP's VQ layer prevents gradients from reaching its last spatial/temporal blocks through the exported global embedding, as verified by smoke testing. Its LoRA and partial runs therefore adapt the final visual projection; this is a model-specific capacity difference that must be reported. Full fine-tuning remains optional and is not scheduled.

The runner computes exact full-training-fold Cox gradients in two deterministic passes: first all scores without encoder graphs, then one recomputed image score at a time weighted by the derivative of the full risk-set loss. This keeps encoder graph memory bounded without changing the loss. It records trainable and total parameters, peak GPU memory, epoch loss, elapsed time, held-out predictions and metrics. All eight one-patient forward/backward checks passed with nonzero gradients in the selected trainable encoder component.

The secondary queue is waiting for the full input cache, then will start automatically on a separate V100 with direct W&B logging:

```bash
WANDB_MODE=online CUDA_VISIBLE_DEVICES=2 bash scripts/queue_lung1_e2_encoder_gpu.sh
```

Logs will be at `logs/lung1_e2/<method>_<model>.log`. `outputs/lung1_e2_encoder_adaptation/COMPLETE` appears only after all eight runs and a refreshed paired comparison finish. No LoRA or partial result is claimed before then. These fixed-hyperparameter secondary rungs are not tuned through inner folds; they use the same locked outer folds and avoid test-set tuning. Their exact raw-image training is substantially more expensive than the cached-feature runs.

Check all queue counters at any time with `.venv/bin/python scripts/status_lung1_e2.py`. The input and token columns should reach 416; each fitted rung should reach 25 folds per model.

## Validation and limitations

The Efron tied-event loss was checked against a hand-calculated tied-death example, the two-pass risk gradient against direct Cox differentiation, and the SQA forward/training path on synthetic token data. `pytest` is not installed in this `.venv`, so the three checks in `tests/test_lung1_e2.py` were run directly. One-patient GPU token extraction and image-level adaptation gradient checks passed for each FM. Full SQA and encoder adaptation require monitoring because those queues had not finished at the time of this record.

CT-CLIP retains the E1 qualification that its checkpoint came from a public mirror and has not been verified identical to the gated author checkpoint. The E2 comparison is internal LUNG1 validation; it does not establish external prognostic transfer.

## BF16 Lightning five-fold rerun (5 October 2026)

The earlier FP32 encoder-adaptation queue was stopped after one TANGERINE LoRA fold. Its results remain under `outputs/lung1_e2_encoder_adaptation` and are not mixed into the new run.

`scripts/run_lung1_e2_encoder_bf16_lightning.py` uses PyTorch Lightning with `bf16-mixed` precision on the A100. It uses only repetition zero of the locked five outer folds. Within each outer training set, inner fold zero is held out for true validation; the other two inner folds train the model for 10 fixed epochs. Thus each fold has about 221 training patients, 111 validation patients, and 83–84 untouched test patients. This faster training split differs from the original full-outer-training-set FP32 protocol, so the results require that qualification in comparisons.

Each epoch logs training and validation Efron Cox losses, Uno C at 36 months, two-year AUC, and integrated Brier score. After the fixed epochs, the outer test pass logs the full survival metric set. One patient CT is forwarded at a time; the full training risk set supplies the exact Cox gradient in two passes. Inputs are BF16 on the GPU, while master parameters and the Cox loss remain float32. Fold predictions are saved under `outputs/lung1_e2_encoder_bf16_lightning/<method>/<model>/folds/`, then combined into `out_of_fold_predictions.csv`. After all five folds, `summary.json` includes pooled metrics over 416 unique patients and the mean fold metrics. Pooled Uno C ranks patients using 36-month cumulative hazard derived from fold-specific survival curves, so arbitrary shifts in raw Cox scores across fitted folds do not affect the ranking.

All eight model/method BF16 forward/backward checks passed on the A100. A complete one-epoch Rad-JEPA-3D LoRA fold check verified Lightning training, validation, test logging, and saved predictions; the local W&B record contains all requested metric keys. Automatic approval review initially rejected direct online W&B synchronization because it would upload internal metrics and run metadata to an external service. After the user explicitly authorized uploading scalar E2 metrics and run metadata, the completed TANGERINE LoRA fold-zero offline run was synced to W&B. The queue was then resumed with `WANDB_MODE=online` from fold one, preserving completed predictions. CT images and patient-level predictions stay local.

The serial queue runs LoRA for TANGERINE, Rad-JEPA-3D, COLIPRI, and CT-CLIP, then partial fine-tuning in the same order. It was started on the A100 using its GPU UUID to avoid a mismatch between CUDA and `nvidia-smi` GPU indices:

```bash
CUDA_VISIBLE_DEVICES=GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01 WANDB_MODE=online \
  bash scripts/launch_lung1_e2_encoder_bf16_lightning.sh
```

Check fold counts with `.venv/bin/python scripts/status_lung1_e2.py`. Per-epoch lines appear in `logs/lung1_e2_bf16/<method>_<model>.log`. Each fold has its own local W&B run in the corresponding output directory. `outputs/lung1_e2_encoder_bf16_lightning/COMPLETE` appears only after all 40 fits finish.
