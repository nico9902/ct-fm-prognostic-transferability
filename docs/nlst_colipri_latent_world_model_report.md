# COLIPRI latent world-model baseline

## Executive result

The first frozen-COLIPRI next-state predictor is working as a latent
forecasting model, but it only recovers a small fraction of the prognostic
advantage present in the actual future state.

On the pooled test pairs, the learned prediction improves on the current state
for the three primary post-t+1 cancer endpoints:

| Endpoint | `z_t` AUROC | `z_hat_t1` AUROC | `z_t1` AUROC | learned − current | learned / actual gap |
|---|---:|---:|---:|---:|---:|
| 1y | 0.714 ± 0.015 | 0.718 ± 0.014 | 0.876 ± 0.004 | +0.004 | −0.158 |
| 2y | 0.658 ± 0.009 | 0.668 ± 0.011 | 0.752 ± 0.011 | +0.010 | −0.084 |
| 3y | 0.643 ± 0.008 | 0.654 ± 0.010 | 0.671 ± 0.014 | +0.011 | −0.017 |

The same pattern is visible in AUPRC. The predicted state is therefore
slightly more prognostic than the present state, especially at 2y/3y, but it
is much less informative than the observed future state at 1y and 2y. At 3y,
it approaches the actual future-state AUROC, although the absolute
discrimination is modest for all representations.

These are mean ± sample SD over five simple probe seeds. The latent predictor
itself was also trained with five seeds; `z_hat_t1` in the comparison tables is
the mean of the five best-validation-seed predictions. The primary endpoint is
the existing landmark-safe label whose outcome window starts after t+1.

## Experimental setup

- Frozen mean-pooled COLIPRI embeddings, 864 dimensions; no COLIPRI
  fine-tuning.
- Existing patient split reused verbatim: 599 train, 200 validation, 200
  test patients.
- 1,984 valid longitudinal pairs: 995 T0→T1 and 989 T1→T2. A patient can
  contribute at most two pairs; missing one embedded scan removes only that
  transition, not the patient.
- Predictor: `z_hat_t1 = z_t + delta_hat`, with LayerNorm on `z_t`, a 1→32
  normalized-time MLP, concatenation, and 1024→1024→864 GELU/dropout MLP.
- Training objective: equal-weight MSE plus cosine loss.
- Δt normalization and every downstream feature scaler were fit on training
  patients only.
- Five model seeds, validation early stopping, AdamW, dropout 0.1.
- W&B project: `nlst-colipri-latent-world-model`.

## Latent forecasting quality

All values below are test-pair means; `learned` is the mean prediction over
the five trained models.

| Scope | Representation | MSE ↓ | Cosine similarity ↑ | L2 distance ↓ |
|---|---|---:|---:|---:|
| pooled | identity (`z_t`) | 0.000965 | 0.975268 | 0.856266 |
| pooled | learned (`z_hat_t1`) | 0.001114 | 0.977884 | 0.937976 |
| T0→T1 | identity | 0.000914 | 0.976851 | 0.844051 |
| T0→T1 | learned | 0.001043 | 0.979798 | 0.917729 |
| T1→T2 | identity | 0.001017 | 0.973677 | 0.868544 |
| T1→T2 | learned | 0.001186 | 0.975961 | 0.958325 |

The identity predictor is stronger on MSE and L2 distance. The learned model
has slightly higher cosine similarity, indicating that it improves directional
alignment while making a larger magnitude error. This is an important warning:
the model is not yet a uniformly better latent reconstruction than simply
copying the current state. The oracle actual future state would have MSE/L2
zero and cosine similarity one by definition.

The transition is harder for T1→T2 than T0→T1 under MSE/L2, consistent with
the later transition being less predictable from the source state in this
cohort.

## Prognostic comparison

The full per-seed table is in
`outputs/colipri_latent_world_model/prognostic_per_seed_metrics.csv`.

### Pooled pairs

| Horizon | Representation | AUROC | AUPRC | MCC | F1 |
|---|---|---:|---:|---:|---:|
| 1y | `z_t` | 0.714 ± 0.015 | 0.175 ± 0.015 | 0.174 ± 0.028 | 0.222 ± 0.016 |
| 1y | `z_hat_t1` | 0.718 ± 0.014 | 0.184 ± 0.010 | 0.166 ± 0.038 | 0.218 ± 0.021 |
| 1y | `z_t1` | 0.876 ± 0.004 | 0.587 ± 0.031 | 0.330 ± 0.006 | 0.336 ± 0.013 |
| 2y | `z_t` | 0.658 ± 0.009 | 0.298 ± 0.011 | 0.177 ± 0.020 | 0.375 ± 0.012 |
| 2y | `z_hat_t1` | 0.668 ± 0.011 | 0.316 ± 0.008 | 0.201 ± 0.010 | 0.389 ± 0.006 |
| 2y | `z_t1` | 0.752 ± 0.011 | 0.484 ± 0.011 | 0.272 ± 0.020 | 0.435 ± 0.013 |
| 3y | `z_t` | 0.643 ± 0.008 | 0.413 ± 0.004 | 0.194 ± 0.018 | 0.467 ± 0.011 |
| 3y | `z_hat_t1` | 0.654 ± 0.010 | 0.425 ± 0.002 | 0.222 ± 0.018 | 0.485 ± 0.009 |
| 3y | `z_t1` | 0.671 ± 0.014 | 0.477 ± 0.011 | 0.207 ± 0.020 | 0.466 ± 0.011 |

At 1y, the learned model's AUROC/AUPRC gain is small and its MCC/F1 are not
better. At 2y and 3y, the learned state improves all four listed metrics over
`z_t`, but remains below `z_t1`, especially at 1y/2y. Thus the main question
has a positive but weak answer for this first baseline:

> `prognosis(z_hat_t1) > prognosis(z_t)` directionally, but
> `prognosis(z_hat_t1)` is not close to `prognosis(z_t1)` at short horizons.

### Pair-type analysis

The split-specific results are noisier because the test subsets contain only
199 T0→T1 and 198 T1→T2 pairs. The stable signal is:

- T0→T1: learned AUROC exceeds current at 2y (0.674 vs 0.658) and 3y
  (0.646 vs 0.631), while the 1y estimate is unstable (0.535 ± 0.157).
- T1→T2: learned AUROC exceeds current at all horizons: 0.724 vs 0.715,
  0.647 vs 0.639, and 0.654 vs 0.642.
- Actual `z_t1` remains substantially stronger than both learned and current
  states, particularly for T1→T2 at 1y/2y.

These subset results should be treated as descriptive, not as definitive
transition-specific claims; the five-seed probe SD is large for some of the
small 1y comparisons.

## Diagnostic-state recovery

The completed native-label diagnostic table is in
`diagnostic_state_per_seed_metrics.csv`. Here `z_t` is evaluated against the
current scan's diagnostic labels, while `z_hat_t1` and `z_t1` are evaluated
against the next scan's labels. This reflects each representation's native
state, but it is not the strictest same-label comparison.

On the pooled pairs:

| Target | `z_t` AUROC | `z_hat_t1` AUROC | `z_t1` AUROC | `z_t` AUPRC | `z_hat_t1` AUPRC | `z_t1` AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| isolation_suspicious | 0.746 | 0.702 | 0.790 | 0.576 | 0.554 | 0.658 |
| nodule_ge_4mm | 0.757 | 0.706 | 0.786 | 0.583 | 0.550 | 0.651 |
| emphysema | 0.717 | 0.753 | 0.743 | 0.615 | 0.704 | 0.681 |
| fibrosis_or_scarring | 0.662 | 0.688 | 0.698 | 0.414 | 0.503 | 0.501 |

The prediction recovers some future-state diagnostic information: it is close
to or above the observed future state for emphysema and has similar AUPRC to
the observed future state for fibrosis/scarring. It underperforms the actual
future state for isolation and nodules, where the observed t+1 scan has a
clear advantage. The apparent emphysema advantage should be validated with a
strict common-label analysis and confidence intervals before interpretation.

## Interpretation

1. The MLP learned a useful transition direction, but its value is not coming
   from better pointwise reconstruction. Identity wins MSE/L2, while the MLP
   wins cosine similarity and downstream pooled prognosis modestly.
2. The predicted state does not simply become an interpolated copy of the
   actual future state in a way that preserves all future diagnostic content.
   The short-horizon cancer signal is the clearest remaining gap.
3. The future-state oracle gap is largest where the outcome is closest to the
   future scan. This is consistent with the model missing localized or
   diagnosis-adjacent changes that are visible in the actual t+1 embedding.
4. The current analysis averages predictions across model seeds for probing.
   The per-seed prediction files and checkpoints are retained for paired
   uncertainty analyses.

## Recommended next steps

### 1. Fix the evaluation protocol before model iteration

Complete the strict diagnostic evaluation in which all three representations
are scored against the same t+1 diagnostic labels, and report paired bootstrap
confidence intervals over test patients. Also report patient-clustered
bootstrap intervals because a patient can contribute two transitions.

### 2. Add a stronger latent objective

The equal MSE+cosine objective does not improve pointwise distance. Compare:

- train-only standardized latent targets and weighted MSE/cosine;
- normalized-vector loss plus a separate norm loss;
- a residual penalty or shrinkage toward identity;
- direct delta prediction with `z_t1 - z_t` as target.

All choices should be selected only on validation patients and compared with
identity on the untouched test patients.

### 3. Improve temporal conditioning without expanding the model much

Run an ablation for no-time, scalar normalized time, log-time, and a small
Fourier/time embedding. The cohort is small, so this should remain a small
number of pre-registered variants rather than a broad hyperparameter search.

### 4. Evaluate individual seed predictions

The current report uses the mean of five model predictions for the downstream
probe. Repeat the prognostic and diagnostic probes for every model seed and
report the distribution of gains over identity. This separates model
uncertainty from probe-seed uncertainty and avoids any ambiguity about seed
ensembling.

### 5. Add clinically meaningful baselines

Compare against a linear delta model, a time-only model, a constant mean
future embedding, and a nearest-neighbor future-state baseline. These will
show whether the nonlinear MLP is learning beyond temporal persistence and
cohort-average drift.

### 6. Test progression-aware objectives

The next practical model should predict a low-dimensional PCA target or a
learned-but-frozen target basis, alongside the full latent target. A useful
variant is a multi-task loss combining latent forecasting with the frozen
prognostic and diagnostic probe directions, while keeping COLIPRI frozen and
training only the transition model. This must be validated carefully to avoid
using test labels to shape the predictor.

### 7. Report longitudinal generalization explicitly

Keep pooled, T0→T1, and T1→T2 results separate in every future experiment.
The current T0→T1/T1→T2 subsets are small, and transition length and label
prevalence should be shown alongside each metric.

## Reproducibility artifacts

- Script: `scripts/run_colipri_latent_world_model.py`
- Pair manifest: `outputs/colipri_latent_world_model/pairs_manifest.csv`
- Checkpoints: `outputs/colipri_latent_world_model/checkpoints/`
- Per-seed predicted embeddings: `outputs/colipri_latent_world_model/predicted_embeddings/`
- Latent distances: `latent_distance_metrics.csv`
- Prognostic probes: `prognostic_per_seed_metrics.csv`
- Diagnostic probes: `diagnostic_state_per_seed_metrics.csv`
- Training histories: `training_history/`
- W&B local/offline cache and synced runs: `wandb/`

