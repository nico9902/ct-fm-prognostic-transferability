# Next Steps

## Research Objective

The goal is to build a latent world model for longitudinal lung CT screening.
RadJEPA3D is used as a frozen 3D CT encoder, and the predictor learns temporal
transitions in latent space:

```text
z_t -> z_{t+1}
```

The paper should not be framed as a standard cancer classifier. The core claim
is that longitudinal CT contains learnable latent dynamics, and that predicted
future latents may carry clinically useful signal for cancer risk or
progression.

## Scale To 2000 Balanced Patients

The next dataset scale is:

```text
tag: pilot_2000_balanced
patients: 2000
sampling: cancer-balanced
expected balance: 1000 cancer_label=1, 1000 cancer_label=0
split: patient-level train/validation/test
```

Prepared commands:

```bash
# 1. Build/download the 2000-patient balanced cohort, convert DICOM to NIfTI,
#    and prepare RadJEPA3D .npy volumes.
bash scripts/run_server_pilot_2000_balanced.sh

# 2. Extract full RadJEPA3D token sequences.
bash scripts/run_server_extract_radjepa_tokens_pilot_2000.sh

# 3. Build token transition table and validate cohort balance + patient split.
bash scripts/run_server_build_token_transitions_pilot_2000.sh

# 4. Train the current main model, Transformer token-wise with CLS output.
bash scripts/run_server_train_transformer_token_cls_z_only_pilot_2000_gpu1.sh

# 5. Run downstream multiseed evaluation.
bash scripts/nohup_run_server_downstream_transformer_token_cls_z_only_pilot_2000_gpu1.sh
```

Expected outputs:

```text
/ssd2/domenico/datasets/nlst/manifests/nlst_pilot_2000_balanced_series.csv
/ssd2/domenico/datasets/nlst/manifests/nlst_pilot_2000_balanced_subjects.csv
/ssd2/domenico/datasets/nlst/manifests/nlst_pilot_2000_balanced_pairs.csv
/ssd2/domenico/datasets/nlst/processed/nlst/pilot_2000_balanced/nii_manifest.csv
/ssd2/domenico/datasets/nlst/processed/nlst/pilot_2000_balanced/radjepa_npy_manifest.csv
/ssd2/domenico/datasets/nlst/embeddings/radjepa3d_pilot_2000_balanced_tokens_manifest.csv
/ssd2/domenico/datasets/nlst/processed/nlst/pilot_2000_balanced/radjepa_token_transitions.csv
outputs/latent_predictor_pilot_2000_balanced_transformer_token_cls_z_only_gpu1/
outputs/downstream_pilot_2000_balanced_transformer_token_cls_z_only_multiseed_gpu1/
```

Validation criteria before interpreting model results:

```text
patients == 2000
cancer_label balance == 1000/1000
no patient overlap between train, validation, and test
missing_or_error_nifti == 0 or manually reviewed
```

## Current Evidence

The 500-patient balanced NLST pilot shows that the latent predictor beats the
persistence baseline:

```text
persistence baseline: z_pred = z_t
model:                z_pred = f(z_t, ...)
```

The strongest ablation so far is `z_only`:

```text
z_only test/loss improvement over persistence:   17.94%
z_only test/l1 improvement over persistence:     14.23%
z_only test/cosine improvement over persistence: 25.82%
```

This suggests that RadJEPA3D CT embeddings already contain enough temporal
signal to learn a useful latent transition without automatic clinical
conditioning.

## 1. Lock the Current Baseline

- Use `z_only` as the main baseline for the next round.
- Keep the 500-patient balanced cohort as the first reproducible pilot.
- Report test improvement over persistence as the primary sanity-check metric.
- Store each run under `outputs/latent_predictor_<tag>_<experiment>_gpu<id>/`.

Primary metrics:

```text
test/loss
test/l1
test/cosine
test/relative_loss_improvement
test/relative_l1_improvement
test/relative_cosine_improvement
```

Interpretation:

- positive relative improvement means the predictor beats persistence;
- negative relative improvement means persistence is still better;
- the current `z_only` result is strong enough to justify downstream clinical
  evaluation.

## 2. Run Small Ablations

Run four ablations on the same patient-level train/validation/test split:

```text
full         = z_t + clinical + delta_t
no_clinical  = z_t + delta_t
no_delta     = z_t + clinical
z_only       = z_t
```

Current conclusion from the first ablation run:

```text
z_only > no_clinical > no_delta ~= full
```

This implies:

- automatic clinical features are not helping yet;
- raw `delta_t` is not helping yet;
- the clean TC-only latent transition is the best current baseline.

Server command:

```bash
bash scripts/nohup_run_server_ablation_predictors_gpu1.sh
```

This runs, in order:

- `full`: `z_t + clinical + delta_t`;
- `no_clinical`: `z_t + delta_t`;
- `no_delta`: `z_t + clinical`;
- `z_only`: `z_t`.

The combined report is written to:

```text
outputs/ablation_pilot_500_balanced_gpu1/ablation_summary.xlsx
```

## 3. Save Test Predictions

For every experiment, save held-out test predictions:

```text
patient_id
source_timepoint
target_timepoint
source_series_uid
target_series_uid
source_embedding_path
target_embedding_path
pred_embedding_path
test_l1
test_cosine
persistence_l1
persistence_cosine
relative_l1_improvement
relative_cosine_improvement
```

These files are needed for downstream clinical evaluation without rerunning the
predictor.

Expected outputs:

```text
outputs/<run_name>/test_predictions.csv
outputs/<run_name>/test_predictions/*.npy
outputs/<run_name>/experiment_report.xlsx
```

## 4. Add Clinical Downstream Tasks

Train simple downstream models to test whether the predicted future latent is
clinically informative.

Compare:

```text
z_t       = current CT latent
z_pred    = predicted future latent
z_{t+1}   = real future CT latent, used as an upper reference
delta_z   = z_pred - z_t
```

Candidate outcomes:

```text
cancer_label
future cancer diagnosis within fixed horizon
future screen-positive result
lesion/progression proxy if reliable fields are available
```

Metrics:

```text
AUROC
AUPRC
balanced accuracy
sensitivity
specificity
calibration if probabilities are used
```

Key paper question:

```text
Does z_pred improve downstream prediction over z_t?
```

If yes, the world model is not just predicting latent similarity; it is moving
the representation toward clinically meaningful future states.

Prepared scripts:

```bash
# Export held-out test predictions from already trained ablation checkpoints.
bash scripts/run_server_export_ablation_predictions.sh

# Run the first downstream task using the best current predictor, z_only.
bash scripts/run_server_downstream_z_only_gpu2.sh

# Same downstream task in background.
bash scripts/nohup_run_server_downstream_z_only_gpu2.sh

# Multi-seed downstream stability check.
bash scripts/nohup_run_server_downstream_z_only_multiseed_gpu2.sh
```

The downstream script trains logistic-regression probes with class balancing and
standardization. It compares:

```text
source_z_t   = current CT latent
pred_z_t1    = predicted future latent
target_z_t1  = real future latent upper reference
pred_delta   = z_pred - z_t
true_delta   = z_{t+1} - z_t
```

Default first outcome:

```text
cancer_label
```

Expected outputs:

```text
outputs/downstream_pilot_500_balanced_z_only_gpu2/downstream_results.csv
outputs/downstream_pilot_500_balanced_z_only_gpu2/downstream_results.xlsx
outputs/downstream_pilot_500_balanced_z_only_gpu2/downstream_metadata.json
```

Main readout:

```text
Compare test_AUROC and test_AUPRC for source_z_t vs pred_z_t1.
```

Paper-relevant positive result:

```text
pred_z_t1 > source_z_t
```

Upper-reference check:

```text
target_z_t1
```

If `target_z_t1` is much stronger than `pred_z_t1`, the future CT latent has
clinical signal but the world model still needs a better predictor. If
`pred_z_t1` approaches `target_z_t1`, that is a strong result.

Multi-seed expected outputs:

```text
outputs/downstream_pilot_500_balanced_z_only_multiseed_gpu2/downstream_multiseed_all_results.csv
outputs/downstream_pilot_500_balanced_z_only_multiseed_gpu2/downstream_multiseed_summary.csv
outputs/downstream_pilot_500_balanced_z_only_multiseed_gpu2/downstream_multiseed_summary.xlsx
```

Stability criterion:

```text
mean(delta_test_auroc_vs_source) > 0
mean(delta_test_auprc_vs_source) > 0
```

If the mean advantage of `pred_z_t1` over `source_z_t` stays positive across
seeds, the downstream result becomes much more credible.

## 5. Scale the Cohort

After the 500-patient pilot is stable:

```text
500 balanced   -> current pilot
1000 balanced
2000 balanced
full eligible longitudinal cohort
```

For each scale:

- keep patient-level splits;
- report persistence-relative improvements;
- report downstream clinical metrics;
- check whether the `z_only` advantage is stable.

Success criterion:

```text
The model should keep positive test/relative_loss_improvement as cohort size grows.
```

## 6. Improve the Predictor

The current predictor is a useful MLP baseline, but it should not be the final
paper model.

Next predictor:

```text
residual z_only:
z_pred = z_t + f(z_t)
```

Why:

- longitudinal CT changes are usually small;
- residual prediction is more natural than predicting the full future latent
  from scratch;
- it is directly comparable with the current `z_only` MLP.

Then test:

```text
z_pred = z_t + f(z_t, time_embedding(delta_t))
z_pred = z_t + f(z_t, curated_clinical_baseline)
Z_pred = token_wise_predictor(Z_t)
```

Transformer CLS pooled variant:

```text
tokens = [CLS, z_t, optional clinical_token, optional delta_token]
H = TransformerEncoder(tokens)
z_pred = Linear(LayerNorm(H_CLS))
```

The CLS token is a learnable summary token. During self-attention it can attend
to the current CT latent and to any conditioning tokens. The final CLS state is
used as the predicted future latent. This is still a pooled-latent model, but it
tests whether attention-style conditioning helps before moving to full
RadJEPA token sequences.

Prepared Transformer CLS command:

```bash
bash scripts/run_server_train_transformer_cls_z_only_gpu2.sh
```

After training:

```bash
bash scripts/nohup_run_server_downstream_transformer_cls_z_only_multiseed_gpu2.sh
```

Expected outputs:

```text
outputs/latent_predictor_pilot_500_balanced_transformer_cls_z_only_gpu2/
outputs/downstream_pilot_500_balanced_transformer_cls_z_only_multiseed_gpu2/
```

Transformer token-wise with CLS output:

```text
RadJEPA3D encoder output:
Z_t = [token_1, token_2, ..., token_1024]

Predictor input:
[CLS, token_1, token_2, ..., token_1024]

Transformer output:
H_CLS

Prediction:
z_pred = Linear(LayerNorm(H_CLS))

Training target:
mean_pool(Z_{t+1})
```

This variant uses the full spatial token sequence from RadJEPA as input, but it
keeps the output as a single future latent vector via the class token. It is the
right intermediate step before predicting all future tokens.

Prepared token-wise commands:

```bash
# 1. Extract full RadJEPA token sequences instead of pooled embeddings.
bash scripts/run_server_extract_radjepa_tokens_pilot_500.sh

# 2. Build a transition table pointing to token .npy files.
bash scripts/run_server_build_token_transitions_pilot_500.sh

# 3. Train token-wise Transformer with CLS output.
bash scripts/run_server_train_transformer_token_cls_z_only_gpu2.sh

# 4. Evaluate downstream stability.
bash scripts/nohup_run_server_downstream_transformer_token_cls_z_only_multiseed_gpu2.sh
```

Expected token-wise outputs:

```text
/ssd2/domenico/datasets/nlst/embeddings/radjepa3d_pilot_500_balanced_tokens_manifest.csv
/ssd2/domenico/datasets/nlst/processed/nlst/pilot_500_balanced/radjepa_token_transitions.csv
outputs/latent_predictor_pilot_500_balanced_transformer_token_cls_z_only_gpu2/
outputs/downstream_pilot_500_balanced_transformer_token_cls_z_only_multiseed_gpu2/
```

Acceptance criterion:

```text
transformer_token_cls should beat direct z_only MLP on either:
1. latent test/relative_loss_improvement, or
2. downstream cancer_label AUROC/AUPRC across seeds.
```

Transformer token-wise with token output:

```text
RadJEPA3D encoder output:
Z_t = [token_1, token_2, ..., token_1024]

Predictor input:
[token_1, token_2, ..., token_1024]

Transformer output:
H_1, H_2, ..., H_1024

Prediction:
Z_pred = Linear(LayerNorm(H))
z_pred = mean_pool(Z_pred)

Training target:
Z_{t+1}
```

This variant predicts the future token sequence directly, then applies mean
pooling only after token prediction. It logs pooled metrics for comparison with
earlier runs and token-level metrics to verify that individual RadJEPA tokens
are being predicted.

Prepared commands:

```bash
# Train future-token predictor.
bash scripts/run_server_train_transformer_token_z_only_gpu2.sh

# Run downstream probes after training.
bash scripts/nohup_run_server_downstream_transformer_token_z_only_multiseed_gpu2.sh
```

Expected outputs:

```text
outputs/latent_predictor_pilot_500_balanced_transformer_token_z_only_gpu2/
outputs/downstream_pilot_500_balanced_transformer_token_z_only_multiseed_gpu2/
```

Prepared command:

```bash
bash scripts/run_server_train_residual_z_only_gpu2.sh
```

This writes:

```text
outputs/latent_predictor_pilot_500_balanced_residual_z_only_gpu2/
```

Compare against the current best direct baseline:

```text
outputs/latent_predictor_pilot_500_balanced_z_only_gpu2/
```

Key comparison:

```text
residual_z_only test/relative_loss_improvement
vs
direct_z_only test/relative_loss_improvement
```

The residual predictor should be accepted only if it improves latent test
metrics and does not reduce downstream `cancer_label` AUROC/AUPRC.

After training, run downstream stability:

```bash
bash scripts/nohup_run_server_downstream_residual_z_only_multiseed_gpu2.sh
```

Expected downstream summary:

```text
outputs/downstream_pilot_500_balanced_residual_z_only_multiseed_gpu2/downstream_multiseed_summary.csv
outputs/downstream_pilot_500_balanced_residual_z_only_multiseed_gpu2/downstream_multiseed_summary.xlsx
```

The token-wise predictor should come later, after the pooled-vector baseline and
downstream clinical evaluation are solid.

## 7. Curate Clinical Features

Do not rely on all automatic numeric `nlst_prsn` features for the paper.

Current automatic features include variables that may be:

- categorical codes treated as continuous values;
- downstream or post-diagnosis fields;
- close to outcome leakage;
- noisy for transition prediction.

Create a curated baseline clinical set containing only non-leaky variables
available at or before baseline screening, such as:

```text
age
sex/gender
race
smoking status
pack-years if available
family history if available
BMI if available
screening arm if appropriate
```

Then rerun:

```text
z_only
z_t + curated_clinical
z_t + curated_clinical + time_embedding(delta_t)
```

Clinical features should remain in the paper only if they improve test results
or downstream clinical utility in a controlled, non-leaky setup.

## 8. Paper-Oriented Deliverables

Prepare these artifacts:

```text
Table 1: cohort statistics and split counts
Table 2: latent prediction metrics vs persistence
Table 3: ablation results
Table 4: downstream clinical prediction results
Figure 1: pipeline schematic
Figure 2: latent prediction vs persistence
Figure 3: downstream comparison z_t vs z_pred vs z_{t+1}
Figure 4: scaling results
```

Main paper message if results hold:

```text
A frozen self-supervised 3D CT encoder plus a lightweight latent predictor can
learn non-trivial longitudinal dynamics in lung screening CT, and the predicted
future latent state may improve clinically relevant risk/progression modeling.
```
