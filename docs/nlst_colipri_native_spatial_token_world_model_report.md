# Native 864-d Spatial COLIPRI World Model

## Executive summary

We reran the spatial COLIPRI world-model experiment without PCA. Frozen COLIPRI dense features were reduced only spatially, from `[24, 24, 24, 864]` to 64 tokens of dimension 864. The native-space Transformer improves latent forecasting over the identity predictor, and it clearly outperforms the previous PCA-256 spatial model. However, after mean-pooling the predicted tokens, its prognostic performance is approximately comparable to the previous pooled residual MLP and remains far below the observed future-state oracle.

The main conclusion is therefore mixed:

- PCA-256 was harmful: removing it restores the full native COLIPRI future-state representation and substantially improves the spatial model.
- Native spatial prediction is a valid direction and improves focal diagnostic recovery relative to PCA-256.
- The current Transformer does not yet recover a meaningful additional prognostic advantage beyond the compact pooled MLP.
- The remaining limitation is likely transition uncertainty and/or the downstream global readout, rather than only loss of spatial resolution from PCA.

## Experimental setup

The experiment used the existing frozen COLIPRI features, longitudinal pairs, patient-level split, and downstream labels.

- 2,984 scan embeddings and 1,984 longitudinal pairs.
- 995 T0→T1 pairs and 989 T1→T2 pairs.
- The existing patient-level train/validation/test split was retained.
- COLIPRI was not fine-tuned.
- No cancer, prognostic, or diagnostic labels were used for world-model training.
- Five independent world-model seeds were trained.
- Downstream linear probes used the same test patients, labels, and probe-seed protocol as the earlier experiments.

### Representation and model

For each scan, the frozen dense grid was deterministically block-mean-pooled:

```text
F_t:                 [24, 24, 24, 864]
spatial mean pool:   [4, 4, 4, 864]
flatten:             R_t = [64, 864]
```

The model predicts a standardized token residual:

```text
Delta_R = R_t1 - R_t
Delta_R_norm = (Delta_R - mu_train) / std_train
R_hat_t1 = R_t + inverse_standardize(Transformer(R_t, Delta_t))
```

The Transformer used 64 tokens, dimension 864, three pre-norm blocks, eight attention heads, feed-forward width 1728, dropout 0.1, learned 3D token positions, and a small time-conditioning MLP projected to the token dimension.

```text
R_t [64 x 864] ──┐
                 ├─> positional + time conditioning
                 └─> 3 × pre-norm Transformer blocks
                         └─> normalized residual [64 x 864]
                                   └─> inverse scaling
R_hat_t1 = R_t + Delta_R_hat
```

All residual scaling statistics were fitted on training pairs only.

## Pooling sanity check

The mean of the 64 native tokens was compared with the existing frozen 864-d mean-pooled COLIPRI embedding for every scan.

| Quantity | Result |
|---|---:|
| Mean MSE | 1.32 × 10⁻¹⁶ |
| Maximum MSE | 2.95 × 10⁻¹⁶ |
| Mean cosine similarity | 1.0000000048 |
| Minimum cosine similarity | 0.9999997616 |
| Mean maximum absolute difference | 1.19 × 10⁻⁷ |
| Maximum absolute difference | 3.58 × 10⁻⁷ |

The implementation is therefore numerically equivalent to the original mean-pooled 864-d COLIPRI representation. The native token model and the existing pooled COLIPRI baseline use the same global representation after token averaging.

## Training

Best validation forecast MSE across the five seeds was:

| Seed | Best validation MSE |
|---:|---:|
| 0 | 0.001841 |
| 1 | 0.001841 |
| 2 | 0.001838 |
| 3 | 0.001833 |
| 4 | 0.001834 |

The runs were logged to W&B and checkpoints were saved separately for each seed.

## Latent forecasting

Metrics below are computed in the native token space. The Transformer values are mean ± standard deviation across five world-model seeds; identity and the observed future state are deterministic references.

### Pooled pairs

| Representation | MSE ↓ | Cosine ↑ | L2 ↓ | Delta cosine ↑ |
|---|---:|---:|---:|---:|
| Identity | 0.001961 | 0.959078 | 10.047018 | 0.000000 |
| Native 864-d Transformer | 0.001831 ± 0.000004 | 0.961219 ± 0.000090 | 9.761836 ± 0.011166 | 0.246438 ± 0.004115 |
| Actual future token state | 0.000000 | 1.000000 | 0.000000 | 1.000000 |

The learned model beats identity on all three future-state reconstruction measures. Relative to identity, MSE decreases by approximately 6.6%, L2 decreases by approximately 2.8%, and cosine similarity increases by about 0.00214. The model predicts a nontrivial portion of the true token residual, as shown by delta cosine around 0.246.

### T0→T1 versus T1→T2

| Pair type | Representation | MSE ↓ | Cosine ↑ | L2 ↓ | Delta cosine ↑ |
|---|---|---:|---:|---:|---:|
| T0→T1 | Identity | 0.001934 | 0.959920 | 10.023490 | 0.000000 |
| T0→T1 | Native Transformer | 0.001781 ± 0.000005 | 0.962490 ± 0.000061 | 9.676054 ± 0.012639 | 0.262304 ± 0.003340 |
| T1→T2 | Identity | 0.001988 | 0.958233 | 10.070663 | 0.000000 |
| T1→T2 | Native Transformer | 0.001882 ± 0.000009 | 0.959941 ± 0.000197 | 9.848051 ± 0.021614 | 0.230492 ± 0.006355 |

Forecasting is easier for T0→T1 than T1→T2. The model captures more residual direction for T0→T1, while the later transition remains more difficult despite similar absolute improvements over identity.

## Prognostic evaluation

Predicted tokens were mean-pooled into native 864-d states before probing. Results are means across five probe seeds and, for learned world models, five world-model seeds. The PCA-256 values are from the earlier spatial experiment and are included as the historical comparison.

### AUROC

| Representation | Cancer 1y | Cancer 2y | Cancer 3y |
|---|---:|---:|---:|
| Current z_t | 0.714 | 0.658 | 0.643 |
| Previous pooled MLP | 0.717 | 0.667 | 0.654 |
| PCA-256 spatial Transformer | 0.589 | 0.577 | 0.615 |
| Native 864-d spatial Transformer | 0.716 | 0.668 | 0.652 |
| Actual z_t1 oracle | 0.876 | 0.752 | 0.671 |

### Other prognostic metrics

| Representation | Horizon | AUPRC | MCC | F1 |
|---|---|---:|---:|---:|
| Current z_t | 1y | 0.175 | 0.174 | 0.222 |
| Native Transformer | 1y | 0.187 | 0.182 | 0.227 |
| Actual z_t1 | 1y | 0.587 | 0.330 | 0.336 |
| Current z_t | 2y | 0.298 | 0.177 | 0.375 |
| Native Transformer | 2y | 0.317 | 0.206 | 0.392 |
| Actual z_t1 | 2y | 0.484 | 0.272 | 0.435 |
| Current z_t | 3y | 0.413 | 0.194 | 0.467 |
| Native Transformer | 3y | 0.419 | 0.217 | 0.481 |
| Actual z_t1 | 3y | 0.477 | 0.207 | 0.466 |

The native Transformer is slightly more prognostic than current z_t at all three horizons and is approximately tied with, or marginally below/above depending on metric, the previous pooled MLP. It does not materially close the oracle gap. The largest remaining gap is at one year:

- current z_t AUROC: 0.714
- native predicted state: 0.716
- actual z_t1: 0.876

For T1→T2, the native Transformer reaches approximately 0.731, 0.647, and 0.656 AUROC at 1y, 2y, and 3y, respectively, versus approximately 0.715, 0.639, and 0.642 for current z_t. For T0→T1, gains are less stable, particularly for the rare 1-year endpoint.

## Strict future diagnostic recovery

All representations were evaluated against the same t+1 diagnostic labels. No representation was compared against its own current labels.

### AUROC

| Representation | Isolation suspicious | Nodule ≥4 mm | Emphysema | Fibrosis/scarring |
|---|---:|---:|---:|---:|
| Current z_t | 0.698 | 0.700 | 0.752 | 0.679 |
| Previous pooled MLP | 0.701 | 0.705 | 0.753 | 0.687 |
| PCA-256 spatial Transformer | 0.679 | 0.670 | 0.758 | 0.612 |
| Native 864-d spatial Transformer | 0.706 | 0.712 | 0.753 | 0.678 |
| Actual z_t1 oracle | 0.790 | 0.786 | 0.743 | 0.698 |

The native model improves over PCA-256 for isolation suspicion and nodule detection, and modestly improves over current z_t for both focal targets. The recovery is still small compared with the observed t+1 state. For emphysema, the current and predicted states are already close to or slightly above the observed-state probe, so this target does not show an oracle advantage. Fibrosis is essentially unchanged relative to current z_t and below the pooled MLP.

## Interpretation

### 1. Does removing PCA restore oracle prognostic performance?

It restores the representation space, not the forecast. The sanity check shows that mean-pooling native tokens exactly reproduces the original 864-d COLIPRI embedding. However, the predicted future state still contains only a small fraction of the diagnostic and prognostic information present in the actual future state. Thus PCA was a real bottleneck for the prior spatial experiment, but removing PCA is not sufficient to solve the transition problem.

### 2. Does native spatial prediction beat the pooled MLP and PCA spatial model?

It clearly beats PCA-256 and has better latent forecasting. It does not clearly beat the pooled standardized-delta MLP downstream. The native Transformer and pooled MLP have nearly identical global prognostic performance, indicating that the current downstream probe primarily uses global information that the pooled MLP already predicts adequately.

### 3. Does native space improve future focal information?

Yes, modestly. Native spatial prediction has higher AUROC than PCA-256 for isolation suspicion (0.706 versus 0.679) and nodule ≥4 mm (0.712 versus 0.670), and slightly exceeds current z_t. Nevertheless, it remains well below the actual t+1 state (0.790 and 0.786), so spatial structure is useful but the current transition objective does not recover enough focal change.

### 4. Does native token space justify continuing?

Yes, as an experimental representation. Native tokens eliminate the confirmed PCA information bottleneck and improve latent and focal diagnostic results. The next improvement should focus on what the model is trained to preserve and how predicted tokens are decoded, rather than immediately increasing Transformer size.

## Files and reproducibility

Results are stored under `outputs/colipri_native_spatial_token_world_model/`:

- `native_spatial_tokens.npy`: cached `[2984, 64, 864]` tokens.
- `pooling_equivalence.json`: pooling sanity-check metrics.
- `delta_scaler.npz`, `time_scaler.npz`: training-fitted scaling parameters.
- `checkpoints/seed*/best.ckpt`: best checkpoint for each world-model seed.
- `predictions/seed*.npy`: predicted token states.
- `latent_per_seed_metrics.csv`: per-seed latent forecasting metrics.
- `downstream_per_seed_metrics.csv`: per-seed prognostic and diagnostic metrics.
- `training_per_seed_metrics.csv`: validation results and checkpoint paths.
- `wandb/`: W&B run data.

## Recommended next steps

1. Keep the native 864-d token representation and add a dual objective: standardized residual MSE plus a global-state cosine or MSE term after token averaging. This directly protects the prognostic signal used by the downstream probe.
2. Add a spatially weighted or feature-wise residual objective so small but clinically important focal changes are not dominated by the many relatively stable voxels/tokens.
3. Evaluate a global-plus-local readout: compare mean pooling with attention pooling or fixed clinically motivated regional pooling, while keeping the world-model architecture unchanged.
4. Stratify by time interval and scan-pair quality. The weaker T1→T2 residual cosine suggests that transition heterogeneity or time normalization may be limiting the forecast.
5. Use bootstrap confidence intervals and paired per-patient comparisons for the small gains over identity and pooled MLP; the current differences are not large enough to claim a robust prognostic improvement.
6. Only after these objective and readout tests should model capacity or token-level architectural complexity be increased.
