# LUNG1 phase 3: fine-tuning sensitivity

Declared on 21 October 2026 before running this sweep. This is a sensitivity
analysis after phases 1, 2 and 4, not a new model-selection experiment.
The external cohort and phase 4 frozen models are not used or modified.

COLIPRI and TANGERINE each receive LoRA and partial last-block adaptation.
LoRA learning rates: 1e-5, 1e-4, 1e-3; partial learning rates: 1e-6, 1e-5,
1e-4. Each rate is evaluated at 5 and 20 fixed epochs. This gives 24
configurations and 120 outer-fold fits. Rank 8, alpha 16, weight decay 1e-4,
architecture, input caches and initialization seeds remain those of E2.

Use repetition zero of the five locked folds. The whole outer training set
(former train plus validation, about 332 patients) is used for every update
and the Breslow baseline. No validation checkpoint selection takes place.
The final epoch alone is evaluated on the outer test fold. Each configuration
starts independently from the pretrained encoder, with seed 20261005 + fold.
Training uses BF16 mixed precision and the exact two-pass Efron Cox gradient.

Report all configurations, primarily mean fold Uno C at 36 months, with
pooled survival-calibrated metrics as secondary outputs. Do not select a
winning configuration on these test results or claim an unbiased performance
estimate for the maximum. Comparison against selected E2 is descriptive:
training sample size and checkpoint rule both change. A failure to improve
does not rule out every possible tuning strategy.

Run `scripts/launch_lung1_phase3.sh` with a BF16 CUDA device visible.
Logs: `logs/lung1_phase3/`; results: `outputs/lung1_phase3/`.
Each setting records provenance, trainable checkpoints, fold metrics and OOF
predictions. Identical completed folds are skipped on restart; incompatible
protocols are rejected. W&B remains offline. `COMPLETE` marks full completion;
`FAILED` marks an interrupted queue due to an error.
