# LUNG1 E1/E2: one five-fold CV with validation selection

This protocol supersedes the earlier five-repetition nested CV for the current
comparison. Earlier results remain archived as pilot results and must not be
pooled with these runs. The original-CT cohort has 416 patients. Repetition
zero of the locked patient-stratified fold file supplies five outer test folds.
Within each outer training split, the previously locked inner fold zero is the
single validation set. Every method and encoder uses identical patient IDs:
approximately 221 train, 111 validation, and 83–84 test patients per fold.

The four pretrained encoder checkpoints and their original-CT input paths are
fixed before any outcome analysis. No tuning uses an outer test fold. For all
neural methods, the epoch with the largest validation Uno C at 36 months is
saved and restored before the sole test evaluation. Ties choose the earliest
epoch. Training runs its full predefined budget; validation selects a
checkpoint, not an early stopping threshold. Uno C uses the training split's
censoring distribution. A nonfinite validation Uno C fails the fold.

| Method | Fixed settings | Validation selection |
| --- | --- | --- |
| Frozen Linear Cox | Standardized global embeddings; 32 PCA components; L2 candidates 0.01, 0.1, 1.0 | Best penalty, retaining its train-only fit. Cox has no epoch checkpoint. |
| Frozen MLP Cox | One hidden layer, width 32, dropout 0.1, AdamW 3e-4, weight decay 1e-4; 100 epochs | Best epoch. |
| Frozen-token SQA | 128-dimensional adapter, four queries, dropout 0.1, AdamW 3e-4, weight decay 1e-4; 40 epochs | Best epoch. |
| LoRA | Rank 8, alpha 16, learning rate 1e-4, weight decay 1e-4; 10 epochs | Best epoch. |
| Partial encoder fine-tuning | Predeclared last trainable block, learning rate 1e-5, weight decay 1e-4; 10 epochs | Best epoch. |

PyTorch Lightning uses BF16 mixed precision for neural forward and backward
passes on the A100. Cox risk-set reductions, loss values, survival estimates,
validation selection and metrics stay FP32 or FP64 for numerical stability.
Frozen Linear Cox has no neural training forward pass, so its PCA and Cox
optimization remain FP32/FP64; forcing BF16 arithmetic on Cox risk sets would
change the statistical calculation without providing useful acceleration.
Existing frozen embeddings and spatial tokens are reused; their extraction
precision is recorded by their provenance files and is not retroactively BF16.

Each test patient receives one out-of-fold risk score and survival curve. The
primary report uses pooled metrics across all 416 predictions, with fold
metrics also retained. Across independently fitted Cox models, the pooled Uno
C uses negative log predicted survival at 36 months rather than raw Cox log
risk, whose origin differs by fold. Horizon-specific cumulative hazards are
used for pooled horizon AUC. The per-fold Breslow baseline comes only from
that fold's training split. Individual predictions and local trainable-weight
checkpoints remain on disk. W&B receives scalar training, validation, test and
pooled metrics plus run metadata; CT images and patient-level predictions are
not uploaded.

Run the serial queue on the A100 with its UUID, with W&B online logging:

```bash
CUDA_VISIBLE_DEVICES=GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01 WANDB_MODE=online \
  bash scripts/launch_lung1_bf16_selected.sh
```

Results go under `outputs/lung1_bf16_selected/<method>/<encoder>` and logs
under `logs/lung1_bf16_selected`. The queue writes `COMPLETE` only after all
20 method/encoder combinations finish their five outer folds.
