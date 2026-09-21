# COLIPRI information-preserving adapter battery

## Status and purpose

Implementation date: 2026-09-21. This document records the corrected baseline, new implementation, verification, and execution plan. Results from the new battery are pending; completed results will be written automatically to `outputs/colipri_preserving_adapter/RESULTS.md`.

**Launch verified:** full battery PID `2535343`, physical GPU 3 (NVIDIA A100 80GB), UUID `GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01`. At the 16:31 CEST check, all five native transition runs had finished and `residual864__seed0` adapter training had begun. W&B is syncing. This is a timestamped status snapshot; use `status.json` and `run.log` for current progress.

**GPU smoke passed:** three variants completed both stages and 20 probe evaluations, producing the final report, validation gate and completion marker. Ordered exports contained all 1,984 pairs from 998 longitudinal patients. For the 1-year endpoint, all smoke representations used the same 396 labeled test pairs (199 T0→T1 and 197 T1→T2). The full scan cache has 2,984 scans and its own 999-patient scan cohort; patients without a usable longitudinal pair are not added to the pair evaluation.

The scientific question is whether an adapter can retain the useful information already present in frozen COLIPRI while making longitudinal prediction more useful. Compression, preservation, and SIGReg are evaluated explicitly. COLIPRI is never instantiated for training; all inputs are the existing frozen mean-pooled embeddings.

## What the earlier audit established

The original temporal-adapter downstream experiment had a feature/label alignment bug. Its training exports were shuffled while prognosis labels stayed in manifest order. Those near-chance downstream tables are invalid. Ordered re-export and a matched GPU probe evaluation were subsequently completed using the original checkpoints. All 30 of those checkpoints had SIGReg disabled.

Corrected pooled test AUROC, averaged over seeds:

| Representation | Cancer 1y | Cancer 2y | Cancer 3y |
|---|---:|---:|---:|
| Native current COLIPRI | 0.715 | 0.654 | 0.640 |
| Linear predicted future | 0.711 | 0.671 | 0.655 |
| Raw residual MLP predicted future | 0.706 | 0.667 | 0.655 |
| Adapter direct predicted future | 0.541 | 0.527 | 0.549 |
| Adapter residual predicted future | 0.562 | 0.581 | 0.614 |
| Adapter residual + time predicted future | 0.563 | 0.581 | 0.614 |
| Adapter residual + velocity + time predicted future | 0.679 | 0.599 | 0.628 |
| Native observed future | 0.870 | 0.742 | 0.672 |

The corrected result still shows reduced linear decodability in the adapter states. However, this should not be confused with the original evaluation bug. Every old adapter predictor also lost to identity in its own latent space. Across five seeds, predicted/identity MSE ratios were 3.32 for direct, 1.037 for residual, 1.048 for residual+time, and 1.062 for residual+velocity+time.

The previous adapter and future target were jointly trainable, permitting contraction toward a nearly constant latent state. Very small future MSE and cosine near one were therefore insufficient measures of successful forecasting. The existing `sigreg_loss` function was a moment penalty, not the official SIGReg statistic. Its behavior is retained for backward compatibility and its docstring now states that distinction.

Observed future scans contain information unavailable at the current scan. The actual-future probe is a reference for accessible information, not a guaranteed achievable ceiling for forecasting from the current scan.

## Registered experiment matrix

| Name | Latent size | Adapter | SIGReg weight |
|---|---:|---|---:|
| native | 864 | Frozen train-standardized COLIPRI | 0 |
| residual864 | 864 | Residual, starts at identity | 0 |
| residual864_sigreg | 864 | Same residual adapter | 0.01 |
| bottleneck256 | 256 | Nonlinear encoder/decoder | 0 |
| bottleneck256_sigreg | 256 | Same bottleneck | 0.01 |
| bottleneck512 | 512 | Nonlinear encoder/decoder | 0 |
| bottleneck512_sigreg | 512 | Same bottleneck | 0.01 |

Five independent seeds (0–4) give 30 adapter pretraining runs and 35 transition runs. Five probe seeds are used for each downstream task/representation. Identical controls can reuse a probe only when the full feature arrays have identical hashes. Embeddings from different world-model seeds are not averaged before probing.

The SIGReg weight, anchor weight, and architecture choices above are prespecified starting values, not claimed optima. All seven configurations are evaluated; the test set is not used to tune or choose them.

## Architecture and training

### Stage 1: preserve the frozen representation

Feature-wise mean and standard deviation are fitted once on unique training scans, using the existing patient split. The input is `x = (h - mean_train) / std_train`. This invertible scaling preserves the native 864-dimensional coordinate system while making reconstruction loss and the Gaussian target numerically meaningful. It does not whiten cross-feature correlations.

The full-dimensional adapter is:

```text
                  ┌──────── identity path ──────────────┐
x [864] ──────────┤                                     +── z [864]
                  └─ Linear(864,256) → LN → GELU         │
                     → Linear(256,864) → × gamma=0.1 ───┘

z → Linear(864,512) → GELU → Linear(512,864) → reconstructed x
```

The final adapter layer is initialized to zero, so initially `z=x`. Gamma is fixed. The decoder receives only `z`; it cannot access `x` through a bypass. Its 512-unit hidden layer is itself a reconstruction constraint.

The compressed adapters are `Linear(864,H) → LayerNorm → GELU → Linear(H,L)`, where `(H,L)` is `(256,256)` or `(512,512)`. The wider adapter really has a 512-unit hidden layer rather than an accidental internal 256-dimensional bottleneck. Both use `Linear(L,512) → GELU → Linear(512,864)` decoders. Parameter counts differ between latent sizes, so these are practical compression comparisons rather than parameter-matched architectures.

Training objectives:

```text
full-dimensional:
  preservation = MSE(decoder(z), x) + 0.1 * MSE(z, x)

compressed:
  preservation = MSE(decoder(z), x)

training loss = preservation + lambda_sigreg * SIGReg(z)
```

No outcome labels enter adapter training. Best checkpoint selection uses validation preservation loss, consistently across SIGReg settings. SIGReg is logged separately and included in optimization, but stochastic Gaussianity estimates do not select the checkpoint. Validation projections use a fixed random seed.

### SIGReg formulation

The implementation follows the random-direction characteristic-function statistic in the [official LeWorldModel code](https://github.com/lucas-maes/le-wm/blob/main/module.py). For each unit direction, it compares empirical cosine/sine averages with the standard normal characteristic function, then integrates the squared error using Gaussian-weighted trapezoidal quadrature. The sample-count multiplier is retained. We use 17 frequencies on [0,3] and 256 projections instead of the upstream default of 1,024 to reduce compute.

SIGReg acts directly on learned states, without per-batch centering or whitening inside the loss. It constrains distributional geometry; it does not certify preservation of specific clinical information. The reconstruction and anchor terms serve a different purpose.

### Stage 2: freeze the adapter and forecast

```text
frozen COLIPRI h_t → training-fitted standardization → frozen adapter → z_t
                                                               │
z_t + normalized delta_t → small residual MLP → normalized delta_hat
                                                               │
inverse delta scaling → delta_hat → z_hat_t1 = z_t + delta_hat
                                                               │
frozen decoder → inverse feature scaling → h_hat_t1 [864]
```

The adapter and decoder have `requires_grad=False` and are in evaluation mode. Their states are cached before training the predictor. The predictor cannot change its target representation.

The transition MLP reuses the existing `ResidualDeltaPredictor`: two 256-unit hidden layers, GELU, dropout 0.1, and a small 32-d time MLP. Inputs are current latent plus training-standardized scan interval. Per-feature delta normalization is fitted on training pairs only. The loss is MSE of normalized deltas, with checkpointing on validation normalized-delta MSE.

All variants use identical time conditioning. Velocity is excluded from this battery to isolate adapter effects and avoid the known missing-history/scan-round confound. This is not a test of the repaired velocity formulation.

Both stages use AdamW, learning rate 0.001, weight decay 0.0001, batch size 128, at most 100 epochs, and validation early stopping with patience 15. All GPU execution runs detached in the background, using GPU UUIDs rather than ambiguous CUDA indices. Lightning logs to W&B and local CSV histories.

## Evaluation protocol

The original longitudinal pairs, labels and patient-level splits are checked against the corrected-evaluation manifest. No cohort regeneration or resplitting occurs.

Forecasting metrics are saved for train/validation/test, pooled and each transition, in both learned latent space and decoded native space: MSE, cosine, L2, delta MSE, delta cosine, and variance-weighted R². Every learned space has its own identity baseline. Native decoded predictions are compared with original future COLIPRI features; native identity is the original current embedding. Raw MSE across different learned spaces should not be ranked directly.

For downstream evaluation, each representation receives the same lightweight linear probe:

- latent current, predicted future, actual future;
- decoded current, decoded predicted future, decoded actual future;
- original native current and actual future controls.

Decoded-current and decoded-actual controls distinguish decoder reconstruction loss from forecasting loss. The decoder has no access to the current input during reconstruction of a predicted state.

The probe matches the corrected baseline: training-only StandardScaler, Linear(D,1), weighted BCE, AdamW (0.001 / 0.0001), batch size 128, maximum 100 epochs, validation-loss checkpoint selection, patience 15, and test threshold 0.5. The historical seeded batch permutation convention is retained. Models and scalers are saved with the probe probabilities and pair IDs.

All representations use the same post-t+1 cancer endpoints (1y, 2y, 3y) and strict t+1 diagnostic targets (isolation_suspicious, nodule_ge_4mm, emphysema, fibrosis_or_scarring). Labels never backpropagate into the adapter or world model.

AUROC, AUPRC, MCC and F1 are reported for validation/test. **Transition-specific downstream rows are subsets of predictions from a pooled-trained probe**, not separately retrained transition probes. This provides a consistent within-run check of heterogeneity but differs from some historical reports that retrained within each transition. Endpoint-specific missing-label masks are shared across representations. Single-class subsets receive undefined discrimination metrics rather than invented values.

The T0→T1 1-year test endpoint has only two positives in the corrected cohort, so its AUROC is unstable. Seed standard deviations measure optimization variability; they are not patient-sampling confidence intervals.

### Preservation gate and scientific interpretation

An adapted current state passes the descriptive validation gate for an endpoint if its mean validation AUROC is no more than 0.02 below the native current control. This is an absolute AUROC margin, not “95% of AUROC.” The gate is written per endpoint and does not tune the model or censor the registered test comparisons. It is not a statistical noninferiority test.

The primary interpretation order is:

1. Does the adapted current state retain native current decodability?
2. Does its actual-future state retain native future decodability?
3. Does prediction improve over identity in the same latent space?
4. Does prediction improve prognosis/diagnosis over that adapted current state and the native control?
5. Is a pooled gain also present within transitions?

Large learned-space reconstruction gains alone are not evidence of restored clinical information. A useful result should survive these controls.

## Implementation and verification

New files:

- `src/lung_jepa/models/preserving_adapter.py`: residual/compact preservation modules, faithful characteristic-function SIGReg, frozen-space transition Lightning module.
- `scripts/run_colipri_preserving_adapter.py`: manifest audit, train-only scaling, two-stage training, exports, matched probes, summaries and restart support.
- `configs/model/colipri_preserving_adapter.yaml`: full registered battery.
- `configs/model/colipri_preserving_adapter_smoke.yaml`: short integration battery.
- `scripts/launch_colipri_preserving_adapter.sh`: detached UUID-based launch with duplicate PID guard.
- `tests/test_preserving_adapter.py`: five regression checks.

Reused components include the authoritative COLIPRI pair builder, existing frozen embeddings and labels, temporal MLP components, metric functions, and corrected linear-probe conventions. Historical model implementations remain compatible. The legacy regularizer received only a clarifying docstring.

Passed CPU checks:

- SIGReg matches an independent trapezoid implementation and has finite gradients.
- Gaussian samples receive a lower SIGReg penalty than constant or mean-shifted samples.
- The 864-d residual adapter starts exactly at identity; its decoder has no input bypass.
- Frozen adapter parameters receive no gradients from transition training.
- A shuffled training loader exports features in dataset order.
- Python compilation and launcher shell syntax checks pass.

The background GPU smoke battery additionally exercises both stages, checkpoint reloads, train-only scalers, prediction exports, paired metrics, downstream probing, resumption markers, and automatic reports. Its metrics are implementation checks, not scientific results.

Smoke outputs: `outputs/colipri_preserving_adapter_smoke/`. [Smoke W&B downstream run](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/tz2353jd). [First full residual-adapter W&B run](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/bciwzeo1).

## Artifacts and execution

Full run: `outputs/colipri_preserving_adapter/`

| Artifact | Purpose |
|---|---|
| `run.log`, `pid`, `status.json` | Process identity, stage and progress |
| `config.json`, `provenance.json` | Fixed configuration, source hashes, GPU and split provenance |
| `pairs_manifest.csv`, `scan_manifest.csv` | Ordered identities and labels/splits |
| `scalers.npz` | Feature/time statistics and training scan indices |
| `runs/<variant>__seed*/adapter/` | Adapter checkpoints and local/W&B histories |
| `runs/<variant>__seed*/transition/` | Frozen-target predictor checkpoints and histories |
| `runs/*/scan_states.npz`, `representations.npz` | Cached states, reconstructions, forecasts and pair IDs |
| `runs/*/delta_scaler.npz` | Training-pair delta statistics |
| `runs/*/*distribution.json` | Per-feature spread and variation/total energy diagnostics |
| `probes/` | Matched probe checkpoints, fitted scalers and val/test probabilities |
| `downstream_metrics.csv` | Incrementally persisted downstream results |
| `downstream_summary.csv`, `latent_summary.csv` | Final grouped results |
| `validation_preservation_gate.csv` | Predeclared representation-quality audit |
| `paired_gains_over_identity.csv` | Paired gains by world-model/probe seed |
| `RESULTS.md`, `COMPLETE.json` | Generated result report and completion marker |

W&B project: `nlst-colipri-temporal-adapter`; full group: `preserving-adapter-v1`; smoke group: `preserving-adapter-smoke`. Training uses one run per stage/seed; downstream metrics and summary tables use a separate run.

Restarting the launcher with the same output/configuration skips completed stages and cached probes. A changed configuration is rejected in an existing output directory. Incomplete training stages restart from scratch; completed checkpoints are reused. Previous experiments and the corrected-evaluation outputs are preserved.

```bash
bash scripts/launch_colipri_preserving_adapter.sh GPU-<physical-uuid>
```

No new outcome claims should be made until the full results and preservation gate are available. This document is the design/work report; the generated `RESULTS.md` is the outcome report.
