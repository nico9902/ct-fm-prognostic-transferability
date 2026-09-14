# Current Project Report

Date: 2026-09-14

## Research Goal

The project aims to build a latent world model for longitudinal lung CT
screening. RadJEPA3D is used as a frozen 3D CT encoder. The trainable component
is a predictor that learns temporal transitions in latent space:

```text
z_t -> z_{t+1}
```

The paper should be framed around learnable longitudinal CT dynamics, not as a
standard cancer classifier. The key scientific question is whether a predicted
future latent state, `z_pred`, contains clinically useful signal beyond the
current CT latent, `z_t`.

## Data Pipeline Implemented

The current pipeline supports the NLST pilot cohort and includes:

- download and organization of NLST imaging metadata and DICOM data;
- download of available clinical tables;
- construction of a balanced 500-patient pilot cohort;
- DICOM to NIfTI conversion;
- NIfTI to RadJEPA3D-ready `.npy` volume preparation;
- RadJEPA3D latent extraction;
- construction of longitudinal transition tables;
- training and evaluation of latent predictors;
- export of held-out test predictions;
- downstream clinical probing with logistic regression.

Main scripts prepared:

```text
scripts/build_nlst_manifest.py
scripts/download_nlst_clinical.py
scripts/convert_nlst_dicom_to_nifti.py
scripts/prepare_radjepa_volumes.py
scripts/extract_radjepa_embeddings.py
scripts/build_latent_transition_dataset.py
scripts/train_latent_predictor.py
scripts/run_downstream_clinical_eval.py
scripts/summarize_downstream_multiseed.py
```

The training code lives in `src/` and uses PyTorch Lightning. Metrics and
configuration can be logged to W&B.

## Cohort And Splits

Current experiment setting:

```text
cohort: pilot_500_balanced
train transitions: 547
validation transitions: 178
test transitions: 186
outcome used downstream: cancer_label
```

The split is patient-level, so transitions from the same patient should not leak
across train, validation, and test.

## RadJEPA3D Representation

Two RadJEPA3D output modes are now supported:

```text
pooled: mean-pooled latent vector, shape [D]
tokens: full token sequence, shape [N, D]
```

For the current RadJEPA3D setup:

```text
embedding_dim: 384
max_tokens used by token-wise predictor: 1024
```

The initial MLP experiments used only the pooled latent vector. The new
Transformer token-wise experiment uses the full RadJEPA3D token sequence as
input.

## Predictors Implemented

### MLP Direct Predictor

Input:

```text
z_t
optional clinical features
optional delta_t
```

Output:

```text
z_pred
```

This was the first baseline. It predicts the future latent directly.

### Residual MLP Predictor

Input:

```text
z_t
```

Output:

```text
z_pred = z_t + delta_z
```

This was tested because longitudinal change might be easier to learn as a
residual update. In the current pilot, it performed worse than the direct MLP.

### Transformer Token-wise CLS Predictor

Input:

```text
[CLS], token_1, token_2, ..., token_N
```

Output:

```text
z_pred = head(H_CLS)
```

The model receives the full RadJEPA3D token sequence from the source CT and uses
the encoded class token as the summary representation for predicting the future
latent. The training target is the mean-pooled future token representation, so
the output remains comparable with the previous pooled-latent experiments.

### Transformer Token-wise Predictor

Input:

```text
token_1, token_2, ..., token_N
```

Output:

```text
pred_token_1, pred_token_2, ..., pred_token_N
z_pred = mean_pool(pred_tokens)
```

This variant does not compress the source sequence into a single CLS vector
before prediction. It predicts the future RadJEPA3D token sequence directly and
only then applies mean pooling. The optimization loss combines the pooled latent
loss with a token-level loss, while the standard `test/loss`, `test/l1`, and
`test/cosine` remain pooled metrics for comparison with earlier runs.

Current configuration:

```text
model_arch: transformer_token_cls
prediction_mode: direct
transformer_dim: 384
transformer_layers: 2
transformer_heads: 8
transformer_mlp_dim: 1536
clinical features: not used
delta_t: not used
```

## Latent Prediction Results

Lower `test_loss`, `test_l1`, and `test_cosine` are better. Positive relative
improvement means the predictor beats persistence, where persistence is:

```text
z_pred = z_t
```

| run | test_loss | test_l1 | test_cosine | rel_loss_impr | rel_l1_impr | rel_cosine_impr |
|:--|--:|--:|--:|--:|--:|--:|
| MLP full | 0.0593 | 0.0415 | 0.1772 | 0.1315 | 0.1034 | 0.1909 |
| MLP no_clinical | 0.0565 | 0.0399 | 0.1659 | 0.1733 | 0.1399 | 0.2441 |
| MLP no_delta | 0.0591 | 0.0415 | 0.1759 | 0.1341 | 0.1042 | 0.1976 |
| MLP z_only | 0.0560 | 0.0397 | 0.1628 | 0.1794 | 0.1423 | 0.2582 |
| Residual z_only | 0.0620 | 0.0432 | 0.1882 | 0.0919 | 0.0687 | 0.1402 |
| Transformer token CLS z_only | **0.0546** | **0.0387** | **0.1598** | 0.1539 | 0.1336 | 0.1836 |

Interpretation:

- all main direct predictors beat persistence;
- clinical features and raw `delta_t` did not help in the first MLP ablation;
- the best pooled baseline is `MLP z_only`;
- the residual MLP is worse than the direct MLP;
- the Transformer token-wise CLS predictor is currently the best model on
  absolute latent prediction metrics.

## Downstream Evaluation Setup

Downstream evaluation trains regularized logistic-regression probes over
different latent feature sets. The goal is not to build the final clinical
classifier, but to test whether the world model moves the representation toward
clinically meaningful future states.

Feature sets:

```text
source_z_t   = current CT latent
pred_z_t1    = predicted future latent
target_z_t1  = real future CT latent, upper reference
pred_delta   = pred_z_t1 - source_z_t
true_delta   = target_z_t1 - source_z_t
```

Metrics:

```text
AUROC
AUPRC
balanced accuracy
sensitivity
specificity
```

The main paper-relevant comparison is:

```text
pred_z_t1 vs source_z_t
```

If `pred_z_t1` improves over `source_z_t`, the world model is adding predictive
future-state information beyond the baseline CT latent.

## Downstream Results

### MLP z_only

| feature_set | test_auroc_mean | test_auprc_mean | test_balanced_accuracy_mean | delta_auroc_vs_source | delta_auprc_vs_source |
|:--|--:|--:|--:|--:|--:|
| source_z_t | 0.5541 | 0.5234 | 0.5398 | 0.0000 | 0.0000 |
| pred_z_t1 | 0.5521 | 0.5151 | 0.5337 | -0.0020 | -0.0083 |
| target_z_t1 | 0.6138 | 0.5797 | 0.5856 | 0.0596 | 0.0563 |
| pred_delta | 0.5452 | 0.5106 | 0.5457 | -0.0089 | -0.0128 |
| true_delta | 0.5337 | 0.5098 | 0.5290 | -0.0205 | -0.0136 |

Interpretation: the MLP predicts latents better than persistence, but its
predicted future latent does not improve downstream cancer prediction over the
current CT latent.

### Residual z_only

| feature_set | test_auroc_mean | test_auprc_mean | test_balanced_accuracy_mean | delta_auroc_vs_source | delta_auprc_vs_source |
|:--|--:|--:|--:|--:|--:|
| source_z_t | 0.5541 | 0.5234 | 0.5398 | 0.0000 | 0.0000 |
| pred_z_t1 | 0.5408 | 0.5082 | 0.5317 | -0.0133 | -0.0152 |
| target_z_t1 | 0.6138 | 0.5797 | 0.5856 | 0.0596 | 0.0563 |
| pred_delta | 0.5518 | 0.5310 | 0.5448 | -0.0024 | 0.0077 |
| true_delta | 0.5337 | 0.5098 | 0.5290 | -0.0205 | -0.0136 |

Interpretation: the simple residual formulation is not supported by the current
evidence. It is worse for latent prediction and does not improve `pred_z_t1`
downstream performance.

### Transformer token CLS z_only

| feature_set | test_auroc_mean | test_auprc_mean | test_balanced_accuracy_mean | delta_auroc_vs_source | delta_auprc_vs_source |
|:--|--:|--:|--:|--:|--:|
| source_z_t | 0.5542 | 0.5236 | 0.5398 | 0.0000 | 0.0000 |
| pred_z_t1 | **0.5707** | 0.5220 | **0.5410** | **0.0165** | -0.0016 |
| target_z_t1 | 0.6137 | 0.5797 | 0.5861 | 0.0595 | 0.0561 |
| pred_delta | 0.5415 | 0.5122 | 0.5296 | -0.0127 | -0.0114 |
| true_delta | 0.5338 | 0.5099 | 0.5284 | -0.0204 | -0.0137 |

Interpretation: this is the strongest result so far. The Transformer token-wise
predictor improves downstream AUROC over `source_z_t` by 0.0165. AUPRC is not
improved yet, but this is the first predictor whose future latent appears to add
clinical signal beyond the current CT latent.

## Current Conclusions

The current evidence supports four conclusions:

1. RadJEPA3D latents contain learnable temporal structure in longitudinal lung
   CT.
2. A simple TC-only predictor is stronger than adding the currently selected
   automatic clinical covariates or raw time delta.
3. A residual MLP is not a good architecture in the current setting.
4. Using the full RadJEPA3D token sequence with a Transformer and CLS output is
   better than pooled MLP prediction on latent metrics, and gives the first
   positive downstream AUROC gain over the current CT latent.

The upper reference `target_z_t1` remains clearly stronger than `pred_z_t1`.
This is useful: it indicates that future CT latents contain clinically relevant
signal, but the world model has not yet recovered all of it.

## Important Output Locations

Latent predictor outputs:

```text
outputs/latent_predictor_pilot_500_balanced_z_only_gpu2/
outputs/latent_predictor_pilot_500_balanced_residual_z_only_gpu2/
outputs/latent_predictor_pilot_500_balanced_transformer_token_cls_z_only_gpu2/
```

Downstream multiseed outputs:

```text
outputs/downstream_pilot_500_balanced_z_only_multiseed_gpu2/
outputs/downstream_pilot_500_balanced_residual_z_only_multiseed_gpu2/
outputs/downstream_pilot_500_balanced_transformer_token_cls_z_only_multiseed_gpu2/
```

Transformer token-wise test predictions:

```text
outputs/latent_predictor_pilot_500_balanced_transformer_token_cls_z_only_gpu2/test_predictions.csv
outputs/latent_predictor_pilot_500_balanced_transformer_token_cls_z_only_gpu2/test_predictions/
```

## Recommended Next Steps

Short-term:

1. Keep `MLP z_only` as the pooled baseline.
2. Keep `Transformer token CLS z_only` as the current main model.
3. Scale the cohort to `pilot_2000_balanced` with 2000 patients, balanced
   1000/1000 by `cancer_label`, while preserving patient-level splitting.
4. Train the new `transformer_token` variant that predicts future tokens before
   mean pooling.
5. Run small Transformer ablations:

```text
transformer_layers: 1, 2, 3
dropout: 0.1, 0.2
lr: 1e-3, 3e-4
token_loss_weight: 0.25, 0.5, 1.0
batch_size: 1 or 2 depending on V100 memory
```

6. Repeat downstream multiseed for the best Transformer ablations.
7. Track both latent prediction and downstream performance, because better
   latent loss alone is not sufficient for the paper claim.

Prepared 2000-patient commands:

```bash
bash scripts/run_server_pilot_2000_balanced.sh
bash scripts/run_server_extract_radjepa_tokens_pilot_2000.sh
bash scripts/run_server_build_token_transitions_pilot_2000.sh
bash scripts/run_server_train_transformer_token_cls_z_only_pilot_2000_gpu1.sh
bash scripts/nohup_run_server_downstream_transformer_token_cls_z_only_pilot_2000_gpu1.sh
```

Medium-term:

1. Scale from the 500-patient pilot to a larger balanced cohort.
2. Add a clinically cleaner endpoint, for example future cancer within a fixed
   horizon.
3. Consider time-aware conditioning only after the TC-only Transformer baseline
   is stable.
4. Add confidence intervals or bootstrap intervals for downstream metrics.
5. Prepare figures showing:

```text
latent prediction vs persistence
source_z_t vs pred_z_t1 vs target_z_t1 downstream performance
gap between pred_z_t1 and target_z_t1
```
