# E3 and E5: exploratory fixed-configuration LoRA extension

Prepared 23 October 2026, after inspecting phase 3 and the original external
validation. COLIPRI and TANGERINE only. LoRA learning rate 1e-3, 20 fixed
epochs, rank 8, alpha 16, weight decay 1e-4. Original CT preprocessing and
pretrained weights remain those of E2. BF16, exact two-pass Efron Cox gradient,
AdamW, final epoch only; no checkpoint or hyperparameter selection.

These are exploratory analyses: the configuration was chosen after inspecting
the phase 3 outer test results, and the external cohort has already been
examined in phase 4. Neither analysis is a new untouched confirmation.

## E3: label efficiency

Reuse the exact nested subsets in `outputs/lung1_e3_frozen_linear/subsets.csv`
(and verify these match the deterministic E3 generator). Five locked outer
folds, repetition zero, 10 draws at 10%, 25%, 50%, one draw at 100%.
This means 155 fits per model, 310 overall. Train on approximately 33, 83,
166, 332 patients per fold. Each fit starts independently from pretrained
weights, with initialization seed 20261005 + fold, as in phase 3. Draws vary
patient selection, not the initialization seed. The 100% fits are rerun to
retain explicit E3 provenance; no extra 100% draws.

Only labels in the selected budget enter optimization and the Breslow baseline.
As in frozen E3, the entire outer development cohort supplies censoring weights
for held-out metric calculation only. This does not enter training. Primary
metric: Uno C at 36 months, first averaged across draws within a fold, then
across the five folds. Pooled calibrated metrics remain secondary. Draws are
not independent patient cohorts and must not be treated as 50 independent folds.
Compare primarily with frozen PCA16 E3, with PCA8 as sensitivity, at the same
budget, draw and fold. No claim of confirmatory superiority from this sweep.

Run:

```bash
bash scripts/launch_lung1_e3_lora.sh
```

Outputs: `outputs/lung1_e3_lora/pct*/draw*/lora/{model}/`, including protocol,
subsets, train IDs, checkpoints, fold metrics and OOF predictions. Root-level
`fit_metrics.csv`, `fold_metrics.csv`, `summary.csv` and matched frozen PCA16/PCA8
differences are generated at completion.
Existing completed folds are skipped, incompatible protocols rejected.
Logs: `logs/lung1_e3_lora/`. Root `COMPLETE` or `FAILED` records queue status.

## E5: external NSCLC-Radiogenomics evaluation

Train one adapter per model on all 416 eligible LUNG1 patients, from pretrained
weights, seed 20261005, for 20 epochs. No external labels or images enter
training. Freeze adapters and LUNG1 baseline risk arrays with SHA256 manifests.
Predict only after freezing both models. Drop external outcome columns before
inference, then evaluate saved predictions with hash checks. New output paths
preserve all original phase 4 results.

Use the existing audited external cohort: primary 143 patients with SEG,
secondary all 199 eligible global CTs. Models use full CTs, not the SEG.
Cache original CT inputs with the existing preprocessing, clamping HU below
-1024 before processing as in phase 4. Require patient and CT series matches.
External cache is approximately 9 GB for both models.

Primary metric is Uno C at 36 months with censoring estimated in the external
population, matching phase 4. Report each model's score and its paired
difference against the corresponding frozen PCA16 phase 4 model. Use 2,000
paired patient bootstrap draws, seed 20261023, percentile 95% intervals.
Intervals are exploratory, without a formal superiority claim or multiple-test
significance declaration. Predicted survival curves use the LUNG1-only Breslow
baseline; no external recalibration. Raw risk scores are comparable within each
single external fit. The primary population overlaps the secondary population.

Run:

```bash
bash scripts/launch_lung1_e5_lora.sh
```

Individual stages: `scripts/run_lung1_e5_lora.py --model colipri|tangerine
--stage train|predict|evaluate`. Outputs: `outputs/lung1_e5_lora/{model}/`.
External input cache: `outputs/radiogenomics_lora_inputs/`.
Logs: `logs/lung1_e5_lora/`. Completed frozen training and predictions resume
with provenance checks. Root `COMPLETE` appears only after both evaluations.

## Runtime estimate and preparation status

Measured phase 3 LoRA lr1e-3/20-epoch five-fold fit time:
COLIPRI 3.512 hours, TANGERINE 1.035 hours (wall-clock logs about 3h31 and 1h03).
E3 patient-work multiplier = 10*(0.10+0.25+0.50)+1 = 9.5.
Linear projection: COLIPRI 33.4 h + TANGERINE 9.8 h = 43.2 h.
Budget **45–55 hours** sequentially on an otherwise free A100 to allow for
310 model initializations, test inference and I/O. This is an estimate, not a
benchmark of the new queue.

E5 training projection: (3.512+1.035)/4 = 1.14 h because two fits on 416
patients replace two sets of five fits on ~333. Budget **2–4 hours** including
external preprocessing, inference and CPU bootstrap; preprocessing is not yet
benchmarked. Combined sequential estimate: **47–59 hours**. Hardware contention
can extend these times. Launchers respect CUDA_VISIBLE_DEVICES; default is the
A100 UUID used in phase 3. No training has been launched during preparation.

Preparation validation: Python compilation and shell syntax checks passed.
Four CPU tests passed (`scripts/test_lung1_e3_e5_lora.py`): exact frozen subset
matching and test exclusion; separation of budget-only Breslow fitting from
full-development metric censoring; rejection of modified frozen adapters;
external pairing/bootstrap returning zero differences for identical stand-in
scores on both populations. These tests do not constitute a GPU training or
external image-inference smoke test; those paths reuse the phase 3 training
module and E2/phase 4 preprocessing, with the new orchestration unbenchmarked.


## Hardware amendment: V100 (23 October 2026)

At the user's request, the A100 queue was terminated before its first completed
fold and its output retained. E3 and E5 move to V100 0 (Tesla V100-PCIE-16GB,
UUID GPU-70cb4254-c439-93d4-1f5b-fbfe9d8e3d0c), sequentially. Learning rate,
epochs, subsets, seeds, LoRA configuration and evaluation remain unchanged.

Use FP16 mixed precision (`--precision 16-mixed`) because V100 lacks native
BF16. Cox loss/risk-set reductions remain FP32 and optimizer parameters FP32.
Lightning manages gradient scaling, initial scale 128. An overflow that skips
an optimizer update raises an error rather than silently completing fewer than
20 updates. FP16 is a numerical protocol change; do not pool these runs with
BF16 phase 3 outputs. The new queue reruns every fit in separate output roots:
`outputs/lung1_e3_lora_v100_fp16/`, `outputs/lung1_e5_lora_v100_fp16/`.

Launcher: `bash scripts/launch_lung1_e3_e5_lora_v100_queue.sh`.
Queue log/status/PID: `logs/lung1_e3_e5_lora_v100/`.
Per-fit logs: `logs/lung1_e3_lora_v100_fp16/` and
`logs/lung1_e5_lora_v100_fp16/`. GPU smoke runs are isolated under
`outputs/lung1_v100_smoke_fp16/` and excluded from experiment results.

Validation after the hardware change: all four CPU tests passed. Both models
completed a one-epoch FP16 GPU smoke run on V100 with 33 training and 84 test
patients, finite loss and Uno C, and no skipped optimizer update. Elapsed fit
plus evaluation: COLIPRI 124 seconds (see per-model smoke fold records for
exact timings). These one-epoch checks do not establish stability over every
20-epoch run; the queue fails explicitly on nonfinite training loss or a
gradient overflow.
