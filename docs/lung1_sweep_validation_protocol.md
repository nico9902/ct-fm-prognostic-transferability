# LUNG1: aligned nested cross-validation for all readouts and adaptation methods

**Status: approved by the author on 26 October 2026.**
- **Amendment 1 (26 October 2026, before any result of this protocol was available):** replaces the
  single-validation-split design approved earlier the same day. The queue already running for inner
  fold 0 (`scripts/launch_lung1_sweep_valsel.sh`) is reused unchanged as the inner-fold-0 stage of
  this design.
- **Amendment 2 (26 October 2026, 18:05, before any result of the nested LoRA/partial runs was
  available):** LoRA and partial fine-tuning are no longer selected by nested cross-validation.
  - Every encoder uses one fixed configuration: LoRA learning rate 1e-3 and partial learning rate
    1e-4, both for 20 epochs, trained on the whole development set and evaluated at the final epoch.
  - This configuration had the highest mean test Uno C for COLIPRI and TANGERINE in the preliminary
    fixed-epoch sweep (`docs/lung1_phase3_protocol.md`). It was therefore chosen after those test
    results had been inspected and is **not** a prespecified choice.
  - For COLIPRI and TANGERINE the reported estimates are the selected sweep fits and are optimistic.
  - For Rad-JEPA-3D, CT-CLIP and DINOv3 the configuration is transferred: none of their own results
    were used to choose it.
  - The nested inner-stage LoRA/partial queues (`outputs/lung1_sweep_valsel/`,
    `outputs/lung1_nested/inner{1,2}/`) were stopped incomplete and are not used.
  - New runs for the three remaining encoders: `scripts/launch_lung1_adaptation_fixed.sh`, outputs in
    `outputs/lung1_adaptation_fixed/`. COLIPRI/TANGERINE reuse `outputs/lung1_phase3/lr1e-3_ep20/lora`
    and `lr1e-4_ep20/partial`.
  - The nested selection below still applies to the frozen linear Cox, MLP and SQA readouts.
- **Context:** written after the fixed-epoch sweep (`docs/lung1_phase3_protocol.md`) and the
  validation-selected comparison (`docs/lung1_bf16_selected_protocol.md`) had been inspected.

## Aim

Compare all trainable readouts and adaptation methods under one common rule:

1. hyperparameters are selected by the same 3-fold inner cross-validation within each outer
   development set;
2. the selected model is refitted on the whole development set (about 332 patients);
3. that model is evaluated once on the outer test fold.

This is the rule the frozen linear Cox readout already uses for its penalty. Every method then uses
the same training patients, the same test patients and the same inner folds, and no choice depends
on test data.

## Data and folds

- Cohort: 416 LUNG1 patients (`data/derived/lung1/cohort_with_gtv_volume.csv`, SHA256 `4d715cd5…`).
- Outer folds: repetition 0 of `configs/lung1_e1_folds.csv` (SHA256 `f4dbe3dd…`). These are the same
  test folds as all other internal analyses.
- Inner folds: the three stratified inner folds of `inner_assignments()` in
  `scripts/run_lung1_frozen_cox_e1.py` (seed 20261001; strata are event × early/late stage),
  repetition 0. Inner fold 0 is the validation set of the validation-selected comparison.
  - Inner stage *k* trains on the other two inner folds (about 221 patients) and validates on fold
    *k* (about 111).
  - The earlier PCA16 frozen reference used different inner folds (`run_lung1_e3_frozen_linear.py`),
    so the frozen readouts are refitted here with the common inner folds.

## Methods, candidates and selection

Each score below is validation Uno C at 36 months, with censoring weights from the inner training
patients. A candidate's score is its **mean over the three inner folds**. The candidate with the
highest mean is selected separately for each outer fold, method and encoder.

| Method | Encoders | Candidates | Inner-stage runs | Final refit on about 332 patients |
|---|---|---|---|---|
| Frozen linear Cox, PCA16 and PCA32 | 5 CT/2-D encoders, Pai, Hosny | L2 penalty 0.01, 0.1, 1 | inside `run_lung1_frozen_cox_e1.py` | same script, repetition 0 |
| MLP (width 32, dropout 0.1, lr 3e-4) | as above | epochs 1–100 | 100 epochs, validation every epoch | new run for the selected number of epochs, final epoch |
| SQA (4 queries, 128-d, lr 3e-4) | 5 CT/2-D encoders | epochs 1–40 | 40 epochs, validation every epoch | as MLP |
| LoRA (rank 8, alpha 16) | all five | *Amendment 2:* none; fixed lr 1e-3, 20 epochs | none | fixed configuration on the whole development set |
| Partial fine-tuning, final block | all five | *Amendment 2:* none; fixed lr 1e-4, 20 epochs | none | as LoRA |

### Ties and failures

- Ties are resolved towards the simpler candidate: larger penalty, fewer epochs, smaller learning
  rate.
- A nonfinite validation score excludes that candidate in that inner fold. A candidate missing in any
  inner fold is not eligible.

### Notes on the LoRA and partial runs

- The epoch 5 value is read from the 20-epoch run. With a constant learning rate and a fixed seed per
  outer fold this matches a 5-epoch run apart from GPU nondeterminism.
- The final refits are the fixed-epoch sweep runs, which use the same seeds, A100 and BF16.
- The inner-stage runs also produce test predictions for their own validation-selected epoch. These
  are not used by this protocol.

## Implementation

All runs use A100, BF16 mixed precision (FP32 Cox loss) and seed 20261005 + outer fold.

- `scripts/run_lung1_e2_encoder_bf16_lightning.py --validation-inner-fold k`. The default 0 keeps the
  earlier behaviour.
- `scripts/run_lung1_bf16_selected_heads.py`:
  - `--validation-inner-fold k`, which also stores the per-epoch validation history;
  - `--fixed-trainval --fold-epochs` for the final refits.
- LoRA/partial inner-stage queues:
  - inner fold 0: `scripts/launch_lung1_sweep_valsel.sh`, outputs in `outputs/lung1_sweep_valsel/`;
  - inner folds 1 and 2: `scripts/launch_lung1_nested_inner.sh`, outputs in
    `outputs/lung1_nested/inner{k}/`.
- MLP/SQA inner stages and final refits: `outputs/lung1_nested/heads/`.
- Frozen linear: `outputs/lung1_nested/linear_pca{16,32}/`.
- Selection and evaluation: `scripts/analyze_lung1_nested.py`.

## Outcomes and tests

- **Metric:** mean over the five outer folds of test Uno C at 36 months, as for the frozen reference.
  Pooled Uno C (negative log predicted survival at 36 months) is secondary.
- **Primary comparisons:** every trainable model against the frozen linear PCA16 readout of the same
  encoder. That is 7 MLP + 5 SQA + 5 LoRA + 5 partial = 22 tests (amendment 2). For COLIPRI and
  TANGERINE, the LoRA and partial contrasts are reported with the caveat that their configuration
  was chosen on these test folds.
  - Paired patient bootstrap within fixed outer folds, 1,000 resamples, seed 20261026.
  - Two-sided Wald z with the bootstrap standard error.
  - Holm correction across the 22 tests.
- **Secondary and descriptive:**
  - the same contrasts against PCA32;
  - PCA32 against PCA16;
  - the selected candidate in each outer fold;
  - the inner-validation versus test gap.
- **Interpretation:** intervals are conditional on the fitted models. A nonsignificant difference is
  not evidence of equivalence.

## Limits

- Inner validation sets are about 111 patients. Selection is noisy and can differ between outer
  folds.
- The LoRA and partial grids were defined after the fixed-epoch sweep had been inspected. Their
  upper learning rates lie at the edge of the grid, and the grid is not extended here.
- Amendment 2: the fixed adaptation configuration is test-informed for COLIPRI and TANGERINE.
  CT-CLIP adapts only its final visual projection.
- No external evaluation is part of this protocol.
