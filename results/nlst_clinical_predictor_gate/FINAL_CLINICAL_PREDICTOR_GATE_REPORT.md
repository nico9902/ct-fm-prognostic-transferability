# Clinical future-predictor gate

**Gate outcome: C (tested deterministic family). No meaningful positive gain recovered by the validation-selected predictor; stop enlarging this deterministic forecasting family.**

**Improved embedding reconstruction is not sufficient evidence of clinically meaningful future-state prediction.** All comparisons below distinguish geometric errors from future-risk discrimination. No RSSM, Dreamer, stochastic dynamics, diffusion, JEPA rollout, or full world model was implemented.

## 1. Cohort, exact labels, and reproduced old baseline

The first gate cohort is unchanged: 999 patients, 2984 cached CTs, 1984 pairs; 599/200/200 patient-level splits. Pair order, feature cache checksum, split assignments, and old checkpoint outputs are verified. The same 864-dimensional frozen mean-pooled COLIPRI representation and extraction provenance are used. Targets are the original future_cancer_1y/2y/3y at the t+1 landmark (365/730/1095 days after that scan), exposed as primary_future_cancer_* in the saved pair manifest. Diagnosis on/before t+1 is excluded and inadequate follow-up is missing, never negative. All current-state conditions share this observed-future-scan eligibility; these are not deployable population-risk estimates.

The exact existing objective-A standardized-delta predictor is reloaded for seeds 0–4; original realized-interval predictions match its cached arrays within 1e-5. Its original five-seed ensemble is retained as old_legacy_realized_interval and first-gate G3_old_legacy. These are retrospective legacy references, not eligible primary current-only baselines. The primary old model uses the SAME checkpoints at the configured 365-day forecast query. The fixed query is available before scanning and removes realized future timing from the new predictor input. Historical old checkpoints were trained with realized intervals; this limitation is explicit. A newly trained geometry_only control uses the same prospective query as C1–C4.

## 2. OracleRiskHead

Linear and 128-unit GELU MLP future-only heads were trained first for seeds [42, 123, 2026], with training-future-only StandardScaler and training-only class weights. Each head jointly predicts the existing three horizon targets using masked macro BCE, restoring the best validation-BCE checkpoint. The ensemble family was selected solely by validation two-year AUROC: **linear**. Its three logits are averaged and frozen, including dropout and normalization. The oracle is excluded from predictor optimization; gradients pass through it only to the predicted embedding. The oracle is an evaluator/teacher, not a deployment model. Future-label weighting makes its sigmoid scores uncalibrated probabilities. Per-head seed variation is separately recorded in oracle_per_seed_metrics.csv; the main frozen ensemble is fixed for all predictors.

| kind | validation_2y_auroc |
| --- | --- |
| linear | 0.8400 |
| mlp | 0.8312 |

## 3. Predictor variants and loss isolation

The exact old DeltaPredictor class is reused: LayerNorm, time MLP 1→32→32, and (864+32)→1024→1024→864 MLP with GELU and dropout 0.1. No backbone capacity is added. Residual mode emits standardized delta and reconstructs z_current + delta; direct mode emits standardized future state. Delta and future scalers fit training pairs only. C1–C4 share architecture and checkpoint-selection rules. The state term retains objective A: mean(((predicted_future−true_future)/train_delta_std)²). It is algebraically equivalent to the original standardized-delta MSE. The auxiliary delta term is standardized-delta SmoothL1 with weight 0.25; state weight is 1.0.

| Variant | Clinical term |
| --- | --- |
| C1 | None: state + delta only |
| C2 | BCE distillation from stop-gradient oracle true-future probabilities |
| C3 | Class-weighted masked BCE against true future labels |
| C4 | Distillation + label BCE |

Distillation is unweighted soft-target BCE; label BCE uses training-only class weights. Both mask invalid/missing horizon targets, including prior-cancer exclusions. They never become negatives. The clinical weight is identical for both terms in C4. State and delta errors are algebraically related; the robust auxiliary loss changes weighting, not the information content of the target. A geometry_only control uses the original loss alone; risk_only uses the identical residual backbone with label loss only, providing a capacity and outcome-supervision control rather than an anatomic forecasting claim.

Clinical weights were predeclared as [0.1, 1.0, 5.0]. Each variant/mode selects one weight by mean validation-only 2y frozen-head AUROC across seeds. Predictor early stopping also uses validation 2y AUROC, so the geometry_only retrain control is essential for separating clinical loss from the historical geometric checkpoint criterion. Best checkpoints are restored. The primary clinical candidate is locked before new test evaluation: **direct_C2**, weight **5.0**.

| mode | variant | clinical_weight | name | validation_2y_auroc |
| --- | --- | --- | --- | --- |
| direct | C1 | 0.0000 | direct_C1_w0 | 0.7853 |
| direct | C2 | 5.0000 | direct_C2_w5 | 0.7864 |
| direct | C3 | 0.1000 | direct_C3_w0.1 | 0.7839 |
| direct | C4 | 0.1000 | direct_C4_w0.1 | 0.7827 |
| residual | C1 | 0.0000 | residual_C1_w0 | 0.7832 |
| residual | C2 | 5.0000 | residual_C2_w5 | 0.7848 |
| residual | C3 | 0.1000 | residual_C3_w0.1 | 0.7799 |
| residual | C4 | 0.1000 | residual_C4_w0.1 | 0.7812 |
| residual | geometry_only | 0.0000 | residual_geometry_only_w0 | 0.7827 |
| residual | risk_only | 1.0000 | residual_risk_only_w1 | 0.7762 |

## 4. State versus delta prediction

Both output parameterizations use identical backbone parameter counts and the same clinical losses. Their output scaling and initialization-induced predictions differ; this is an explicit parameterization ablation. Paired residual-minus-direct results are in paired_comparisons.csv. No claim that a delta carries independent information beyond current+future is made.

## 5–7. Identity, geometric quality, and clinical metrics

| model | delta_mse | delta_cosine | mse | cosine |
| --- | --- | --- | --- | --- |
| current | 0.0010 | 0.0000 | 0.0010 | 0.9753 |
| direct_C1 | 0.0014 | 0.3138 | 0.0014 | 0.9634 |
| direct_C2 | 0.0016 | 0.3207 | 0.0016 | 0.9587 |
| direct_C3 | 0.0013 | 0.3192 | 0.0013 | 0.9667 |
| direct_C4 | 0.0013 | 0.3191 | 0.0013 | 0.9665 |
| identity | 0.0010 | 0.0000 | 0.0010 | 0.9753 |
| old | 0.0009 | 0.2899 | 0.0009 | 0.9773 |
| old_legacy_realized_interval | 0.0009 | 0.2876 | 0.0009 | 0.9772 |
| real_future | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| residual_C1 | 0.0009 | 0.2805 | 0.0009 | 0.9769 |
| residual_C2 | 0.0009 | 0.2320 | 0.0009 | 0.9766 |
| residual_C3 | 0.0009 | 0.2451 | 0.0009 | 0.9767 |
| residual_C4 | 0.0009 | 0.2440 | 0.0009 | 0.9767 |
| residual_geometry_only | 0.0009 | 0.2779 | 0.0009 | 0.9769 |
| residual_risk_only | 0.0010 | 0.0690 | 0.0010 | 0.9753 |

| channel | model | head | auroc | auprc | mcc |
| --- | --- | --- | --- | --- | --- |
| direct | G0 | linear | 0.669 ± 0.023 | 0.311 ± 0.018 | 0.178 ± 0.033 |
| direct | G0 | mlp | 0.650 ± 0.005 | 0.299 ± 0.010 | 0.175 ± 0.005 |
| direct | G1 | linear | 0.734 ± 0.004 | 0.469 ± 0.033 | 0.243 ± 0.039 |
| direct | G1 | mlp | 0.735 ± 0.013 | 0.443 ± 0.032 | 0.290 ± 0.018 |
| direct | G2 | linear | 0.739 ± 0.010 | 0.487 ± 0.002 | 0.249 ± 0.006 |
| direct | G2 | mlp | 0.726 ± 0.010 | 0.489 ± 0.015 | 0.238 ± 0.019 |
| direct | G3_old | linear | 0.664 ± 0.008 | 0.314 ± 0.006 | 0.194 ± 0.030 |
| direct | G3_old | mlp | 0.642 ± 0.013 | 0.303 ± 0.018 | 0.190 ± 0.006 |
| direct | G3_old_legacy | linear | 0.663 ± 0.008 | 0.315 ± 0.007 | 0.195 ± 0.033 |
| direct | G3_old_legacy | mlp | 0.644 ± 0.012 | 0.305 ± 0.018 | 0.194 ± 0.009 |
| direct | direct_C1 | linear | 0.646 ± 0.014 | 0.306 ± 0.007 | 0.175 ± 0.024 |
| direct | direct_C1 | mlp | 0.626 ± 0.021 | 0.292 ± 0.029 | 0.150 ± 0.063 |
| direct | direct_C2 | linear | 0.659 ± 0.015 | 0.311 ± 0.022 | 0.171 ± 0.023 |
| direct | direct_C2 | mlp | 0.639 ± 0.015 | 0.296 ± 0.027 | 0.174 ± 0.058 |
| direct | direct_C3 | linear | 0.649 ± 0.021 | 0.316 ± 0.009 | 0.166 ± 0.022 |
| direct | direct_C3 | mlp | 0.634 ± 0.026 | 0.297 ± 0.033 | 0.169 ± 0.050 |
| direct | direct_C4 | linear | 0.648 ± 0.022 | 0.315 ± 0.010 | 0.167 ± 0.013 |
| direct | direct_C4 | mlp | 0.635 ± 0.025 | 0.296 ± 0.033 | 0.166 ± 0.055 |
| direct | residual_C1 | linear | 0.665 ± 0.006 | 0.315 ± 0.006 | 0.186 ± 0.024 |
| direct | residual_C1 | mlp | 0.644 ± 0.013 | 0.303 ± 0.016 | 0.183 ± 0.017 |
| direct | residual_C2 | linear | 0.663 ± 0.007 | 0.307 ± 0.004 | 0.188 ± 0.041 |
| direct | residual_C2 | mlp | 0.640 ± 0.014 | 0.302 ± 0.020 | 0.180 ± 0.013 |
| direct | residual_C3 | linear | 0.661 ± 0.011 | 0.306 ± 0.007 | 0.191 ± 0.033 |
| direct | residual_C3 | mlp | 0.639 ± 0.017 | 0.303 ± 0.020 | 0.184 ± 0.014 |
| direct | residual_C4 | linear | 0.661 ± 0.011 | 0.306 ± 0.007 | 0.191 ± 0.033 |
| direct | residual_C4 | mlp | 0.639 ± 0.017 | 0.303 ± 0.020 | 0.184 ± 0.017 |
| direct | residual_geometry_only | linear | 0.665 ± 0.007 | 0.312 ± 0.003 | 0.191 ± 0.020 |
| direct | residual_geometry_only | mlp | 0.646 ± 0.012 | 0.303 ± 0.016 | 0.179 ± 0.010 |
| direct | residual_risk_only | linear | 0.659 ± 0.012 | 0.302 ± 0.007 | 0.192 ± 0.037 |
| direct | residual_risk_only | mlp | 0.635 ± 0.016 | 0.299 ± 0.019 | 0.164 ± 0.026 |
| risk | current | frozen_oracle | 0.685 ± 0.000 | 0.318 ± 0.000 | 0.254 ± 0.000 |
| risk | direct_C1 | frozen_oracle | 0.647 ± 0.017 | 0.300 ± 0.008 | 0.172 ± 0.028 |
| risk | direct_C2 | frozen_oracle | 0.669 ± 0.003 | 0.310 ± 0.002 | 0.184 ± 0.009 |
| risk | direct_C3 | frozen_oracle | 0.657 ± 0.017 | 0.308 ± 0.007 | 0.191 ± 0.003 |
| risk | direct_C4 | frozen_oracle | 0.654 ± 0.018 | 0.307 ± 0.008 | 0.189 ± 0.008 |
| risk | identity | frozen_oracle | 0.685 ± 0.000 | 0.318 ± 0.000 | 0.254 ± 0.000 |
| risk | old | frozen_oracle | 0.693 ± 0.000 | 0.327 ± 0.000 | 0.253 ± 0.000 |
| risk | old_legacy_realized_interval | frozen_oracle | 0.692 ± 0.000 | 0.330 ± 0.000 | 0.237 ± 0.000 |
| risk | real_future | frozen_oracle | 0.754 ± 0.000 | 0.488 ± 0.000 | 0.289 ± 0.000 |
| risk | residual_C1 | frozen_oracle | 0.692 ± 0.002 | 0.326 ± 0.000 | 0.257 ± 0.005 |
| risk | residual_C2 | frozen_oracle | 0.689 ± 0.005 | 0.320 ± 0.002 | 0.236 ± 0.007 |
| risk | residual_C3 | frozen_oracle | 0.685 ± 0.004 | 0.324 ± 0.006 | 0.226 ± 0.024 |
| risk | residual_C4 | frozen_oracle | 0.685 ± 0.004 | 0.324 ± 0.007 | 0.234 ± 0.028 |
| risk | residual_geometry_only | frozen_oracle | 0.691 ± 0.002 | 0.324 ± 0.005 | 0.250 ± 0.011 |
| risk | residual_risk_only | frozen_oracle | 0.685 ± 0.001 | 0.320 ± 0.002 | 0.248 ± 0.028 |

risk evaluates R(predicted_future) with the SAME frozen head; current/identity use R(current), and real_future is the oracle upper reference. direct evaluates a separately trained [current,predicted_future] linear or 128-unit MLP, with the unchanged first-gate G0/G1/G2 heads reused verbatim. G0 is padded current-only and G1 is current+real future. New probes use training-only feature scaling and validation-best masked BCE, never test-based selection. Train/validation/test AUROC, AUPRC, MCC and additional threshold metrics are in per_seed_metrics.csv and experiment_summary.csv. Threshold 0.5 is fixed.

Metrics are mean ± sample SD across 42,123,2026; predictor variation is included for new models, while frozen-head and legacy ensembles are fixed. Baseline risk SD=0 reflects a fixed ensemble, not absence of uncertainty. Downstream new-model seeds couple predictor and probe variation. Legacy predictors retain their five original seeds, with individual outputs preserved.

## 8. Oracle gap and relative closure

OracleGap = AUROC(R(real future)) − AUROC(R(predicted future)). GapClosure = (AUROC(R(predicted future)) − AUROC(R(current))) / (AUROC(R(real future)) − AUROC(R(current))). The numerator/denominator use the same teacher and eligible samples; the R(current) baseline is distinct from G0, which trains a head on current features for this endpoint. Ratios are reported only for positive denominators. Confidence intervals for ratios can be unstable when the oracle-current gap is small; positive-denominator bootstrap draw counts are supplied. Values can be negative or exceed 100% and are not clipped.

| model | AUC_current_mean | AUC_oracle_mean | AUC_predicted_mean | OracleGap_mean | OracleGap_std | GapClosure_mean | GapClosure_std | ensemble_OracleGap | OracleGap_ci_low | OracleGap_ci_high | ensemble_GapClosure | GapClosure_ci_low | GapClosure_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 0.6849 | 0.7536 | 0.6849 | 0.0687 | 0.0000 | 0.0000 | 0.0000 | 0.0687 | 0.0237 | 0.1141 | 0.0000 | 0.0000 | 0.0000 |
| direct_C1 | 0.6849 | 0.7536 | 0.6472 | 0.1064 | 0.0166 | -0.5493 | 0.2424 | 0.1028 | 0.0458 | 0.1603 | -0.4970 | -1.6667 | 0.0013 |
| direct_C2 | 0.6849 | 0.7536 | 0.6687 | 0.0849 | 0.0033 | -0.2363 | 0.0488 | 0.0827 | 0.0329 | 0.1305 | -0.2051 | -0.8554 | 0.1131 |
| direct_C3 | 0.6849 | 0.7536 | 0.6573 | 0.0963 | 0.0171 | -0.4027 | 0.2496 | 0.0904 | 0.0402 | 0.1403 | -0.3164 | -1.1368 | 0.0398 |
| direct_C4 | 0.6849 | 0.7536 | 0.6542 | 0.0994 | 0.0181 | -0.4471 | 0.2636 | 0.0933 | 0.0426 | 0.1447 | -0.3581 | -1.2263 | 0.0111 |
| identity | 0.6849 | 0.7536 | 0.6849 | 0.0687 | 0.0000 | 0.0000 | 0.0000 | 0.0687 | 0.0237 | 0.1141 | 0.0000 | 0.0000 | 0.0000 |
| old | 0.6849 | 0.7536 | 0.6926 | 0.0610 | 0.0000 | 0.1113 | 0.0000 | 0.0610 | 0.0162 | 0.1058 | 0.1113 | -0.2001 | 0.4339 |
| old_legacy_realized_interval | 0.6849 | 0.7536 | 0.6924 | 0.0612 | 0.0000 | 0.1083 | 0.0000 | 0.0612 | 0.0169 | 0.1056 | 0.1083 | -0.2531 | 0.4304 |
| real_future | 0.6849 | 0.7536 | 0.7536 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| residual_C1 | 0.6849 | 0.7536 | 0.6920 | 0.0616 | 0.0016 | 0.1032 | 0.0237 | 0.0601 | 0.0151 | 0.1052 | 0.1246 | -0.1813 | 0.4590 |
| residual_C2 | 0.6849 | 0.7536 | 0.6891 | 0.0645 | 0.0045 | 0.0603 | 0.0661 | 0.0627 | 0.0175 | 0.1085 | 0.0871 | -0.1232 | 0.3622 |
| residual_C3 | 0.6849 | 0.7536 | 0.6849 | 0.0687 | 0.0038 | -0.0010 | 0.0560 | 0.0671 | 0.0216 | 0.1137 | 0.0224 | -0.3246 | 0.2896 |
| residual_C4 | 0.6849 | 0.7536 | 0.6848 | 0.0688 | 0.0042 | -0.0014 | 0.0617 | 0.0669 | 0.0220 | 0.1130 | 0.0254 | -0.3066 | 0.2838 |
| residual_geometry_only | 0.6849 | 0.7536 | 0.6911 | 0.0625 | 0.0018 | 0.0897 | 0.0264 | 0.0603 | 0.0160 | 0.1057 | 0.1216 | -0.1911 | 0.4678 |
| residual_risk_only | 0.6849 | 0.7536 | 0.6854 | 0.0682 | 0.0013 | 0.0065 | 0.0185 | 0.0670 | 0.0224 | 0.1110 | 0.0242 | -0.1557 | 0.1897 |

## 9. Paired uncertainty

| channel | head | model | comparator | task | metric | gain | ci_low | ci_high | valid_draws | primary_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risk | frozen_oracle | direct_C2 | direct_C1 | primary_future_cancer_2y | auroc | 0.0200 | -0.0006 | 0.0410 | 2000 | True |
| direct | linear | direct_C2 | direct_C1 | primary_future_cancer_2y | auroc | 0.0121 | -0.0053 | 0.0325 | 2000 | True |
| direct | mlp | direct_C2 | direct_C1 | primary_future_cancer_2y | auroc | 0.0186 | 0.0005 | 0.0372 | 2000 | True |
| risk | frozen_oracle | direct_C2 | old | primary_future_cancer_2y | auroc | -0.0217 | -0.0430 | -0.0027 | 2000 | True |
| risk | frozen_oracle | direct_C2 | identity | primary_future_cancer_2y | auroc | -0.0141 | -0.0372 | 0.0086 | 2000 | True |
| risk | frozen_oracle | direct_C2 | current | primary_future_cancer_2y | auroc | -0.0141 | -0.0372 | 0.0086 | 2000 | True |
| risk | frozen_oracle | real_future | direct_C2 | primary_future_cancer_2y | auroc | 0.0827 | 0.0329 | 0.1305 | 2000 | True |
| direct | linear | direct_C2 | G3_old | primary_future_cancer_2y | auroc | -0.0019 | -0.0237 | 0.0198 | 2000 | True |
| direct | linear | direct_C2 | G0 | primary_future_cancer_2y | auroc | -0.0071 | -0.0325 | 0.0170 | 2000 | True |
| direct | linear | G1 | direct_C2 | primary_future_cancer_2y | auroc | 0.0753 | 0.0349 | 0.1170 | 2000 | True |
| direct | mlp | direct_C2 | G3_old | primary_future_cancer_2y | auroc | 0.0013 | -0.0084 | 0.0111 | 2000 | True |
| direct | mlp | direct_C2 | G0 | primary_future_cancer_2y | auroc | -0.0016 | -0.0227 | 0.0196 | 2000 | True |
| direct | mlp | G1 | direct_C2 | primary_future_cancer_2y | auroc | 0.0896 | 0.0416 | 0.1358 | 2000 | True |
| risk | frozen_oracle | direct_C2 | direct_C3 | primary_future_cancer_2y | auroc | 0.0076 | -0.0065 | 0.0243 | 2000 | True |
| direct | linear | direct_C2 | direct_C3 | primary_future_cancer_2y | auroc | 0.0069 | -0.0100 | 0.0250 | 2000 | True |
| direct | mlp | direct_C2 | direct_C3 | primary_future_cancer_2y | auroc | 0.0078 | -0.0059 | 0.0223 | 2000 | True |
| risk | frozen_oracle | direct_C2 | direct_C4 | primary_future_cancer_2y | auroc | 0.0105 | -0.0059 | 0.0294 | 2000 | True |
| direct | linear | direct_C2 | direct_C4 | primary_future_cancer_2y | auroc | 0.0082 | -0.0095 | 0.0272 | 2000 | True |
| direct | mlp | direct_C2 | direct_C4 | primary_future_cancer_2y | auroc | 0.0075 | -0.0063 | 0.0218 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_C1 | primary_future_cancer_2y | auroc | -0.0226 | -0.0439 | -0.0023 | 2000 | True |
| direct | linear | direct_C2 | residual_C1 | primary_future_cancer_2y | auroc | -0.0025 | -0.0241 | 0.0194 | 2000 | True |
| direct | mlp | direct_C2 | residual_C1 | primary_future_cancer_2y | auroc | -0.0007 | -0.0108 | 0.0095 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_C2 | primary_future_cancer_2y | auroc | -0.0201 | -0.0427 | 0.0023 | 2000 | True |
| direct | linear | direct_C2 | residual_C2 | primary_future_cancer_2y | auroc | -0.0013 | -0.0219 | 0.0189 | 2000 | True |
| direct | mlp | direct_C2 | residual_C2 | primary_future_cancer_2y | auroc | 0.0040 | -0.0064 | 0.0146 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_C3 | primary_future_cancer_2y | auroc | -0.0156 | -0.0393 | 0.0065 | 2000 | True |
| direct | linear | direct_C2 | residual_C3 | primary_future_cancer_2y | auroc | 0.0011 | -0.0201 | 0.0214 | 2000 | True |
| direct | mlp | direct_C2 | residual_C3 | primary_future_cancer_2y | auroc | 0.0055 | -0.0052 | 0.0166 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_C4 | primary_future_cancer_2y | auroc | -0.0158 | -0.0393 | 0.0058 | 2000 | True |
| direct | linear | direct_C2 | residual_C4 | primary_future_cancer_2y | auroc | 0.0010 | -0.0200 | 0.0212 | 2000 | True |
| direct | mlp | direct_C2 | residual_C4 | primary_future_cancer_2y | auroc | 0.0054 | -0.0053 | 0.0165 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_geometry_only | primary_future_cancer_2y | auroc | -0.0224 | -0.0453 | -0.0012 | 2000 | True |
| direct | linear | direct_C2 | residual_geometry_only | primary_future_cancer_2y | auroc | -0.0026 | -0.0249 | 0.0200 | 2000 | True |
| direct | mlp | direct_C2 | residual_geometry_only | primary_future_cancer_2y | auroc | -0.0015 | -0.0127 | 0.0093 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_risk_only | primary_future_cancer_2y | auroc | -0.0157 | -0.0405 | 0.0071 | 2000 | True |
| direct | linear | direct_C2 | residual_risk_only | primary_future_cancer_2y | auroc | 0.0021 | -0.0189 | 0.0224 | 2000 | True |
| direct | mlp | direct_C2 | residual_risk_only | primary_future_cancer_2y | auroc | 0.0099 | -0.0018 | 0.0223 | 2000 | True |
| risk | frozen_oracle | residual_C2 | direct_C2 | primary_future_cancer_2y | auroc | 0.0201 | -0.0023 | 0.0427 | 2000 | True |

Bootstrap uses 2000 patient-cluster resamples of the held-out seed-ensemble predictions. All eligible transitions from a sampled patient travel together, and the same draws evaluate every arm. Exact weighted AUROC/AP/MCC is used for speed and tested against explicit sklearn resampling, including ties. Intervals are conditional on fitted models; training-seed SD remains separate. Paired gap reduction equals the candidate-minus-old AUROC difference because both share the oracle. Primary candidate selection used only validation; other model/endpoint comparisons are exploratory without multiplicity correction. Nonsignificance is not equivalence. Meaningful gain is predeclared as AUROC ≥0.02 with paired 95% CI above zero; equivalence requires the whole interval inside ±0.02.

## 10. Subgroups, error analysis, and representations

| model | subgroup | n | patients | positives | auroc | auprc | mcc | mse | clinical_probability_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity | all_test | 396 | 200 | 75 | 0.6849 | 0.3181 | 0.2538 | 0.0010 | 0.1086 |
| identity | T0_T1 | 199 | 199 | 30 | 0.7284 | 0.2813 | 0.2800 | 0.0009 | 0.0937 |
| identity | T1_T2 | 197 | 197 | 45 | 0.6485 | 0.3546 | 0.2154 | 0.0010 | 0.1237 |
| identity | future_cancer_2y_positive | 75 | 48 | 75 | NA | NA | NA | 0.0013 | 0.1651 |
| identity | future_cancer_2y_negative | 321 | 170 | 0 | NA | NA | NA | 0.0009 | 0.0954 |
| identity | future_cancer_within_1y | 31 | 31 | 31 | NA | NA | NA | 0.0019 | 0.2252 |
| identity | future_cancer_after_1y | 165 | 98 | 44 | 0.5582 | 0.3066 | 0.1479 | 0.0009 | 0.1061 |
| identity | current_isolation_suspicious_0 | 270 | 154 | 47 | 0.6864 | 0.2757 | 0.2389 | 0.0010 | 0.1036 |
| identity | current_isolation_suspicious_1 | 126 | 81 | 28 | 0.6939 | 0.4192 | 0.2698 | 0.0009 | 0.1193 |
| identity | current_nodule_ge_4mm_0 | 272 | 154 | 48 | 0.6733 | 0.2732 | 0.2329 | 0.0010 | 0.1037 |
| identity | current_nodule_ge_4mm_1 | 124 | 79 | 27 | 0.7159 | 0.4264 | 0.2914 | 0.0009 | 0.1193 |
| identity | current_emphysema_0 | 251 | 142 | 46 | 0.6794 | 0.3301 | 0.2703 | 0.0009 | 0.1086 |
| identity | current_emphysema_1 | 145 | 90 | 29 | 0.6894 | 0.3185 | 0.2252 | 0.0010 | 0.1085 |
| identity | current_fibrosis_or_scarring_0 | 287 | 163 | 52 | 0.6900 | 0.3174 | 0.2567 | 0.0009 | 0.1083 |
| identity | current_fibrosis_or_scarring_1 | 109 | 73 | 23 | 0.6704 | 0.3459 | 0.2410 | 0.0010 | 0.1095 |
| old | all_test | 396 | 200 | 75 | 0.6926 | 0.3265 | 0.2526 | 0.0009 | 0.1124 |
| old | T0_T1 | 199 | 199 | 30 | 0.7320 | 0.3061 | 0.2750 | 0.0008 | 0.0951 |
| old | T1_T2 | 197 | 197 | 45 | 0.6589 | 0.3581 | 0.2215 | 0.0009 | 0.1298 |
| old | future_cancer_2y_positive | 75 | 48 | 75 | NA | NA | NA | 0.0012 | 0.1546 |
| old | future_cancer_2y_negative | 321 | 170 | 0 | NA | NA | NA | 0.0008 | 0.1025 |
| old | future_cancer_within_1y | 31 | 31 | 31 | NA | NA | NA | 0.0017 | 0.2023 |
| old | future_cancer_after_1y | 165 | 98 | 44 | 0.5594 | 0.3244 | 0.1089 | 0.0008 | 0.1104 |
| old | current_isolation_suspicious_0 | 270 | 154 | 47 | 0.7034 | 0.2947 | 0.2549 | 0.0009 | 0.1077 |
| old | current_isolation_suspicious_1 | 126 | 81 | 28 | 0.6848 | 0.4315 | 0.2340 | 0.0008 | 0.1223 |
| old | current_nodule_ge_4mm_0 | 272 | 154 | 48 | 0.6902 | 0.2917 | 0.2434 | 0.0009 | 0.1079 |
| old | current_nodule_ge_4mm_1 | 124 | 79 | 27 | 0.7060 | 0.4391 | 0.2659 | 0.0009 | 0.1222 |
| old | current_emphysema_0 | 251 | 142 | 46 | 0.6971 | 0.3401 | 0.2705 | 0.0009 | 0.1080 |
| old | current_emphysema_1 | 145 | 90 | 29 | 0.6807 | 0.3253 | 0.2213 | 0.0009 | 0.1199 |
| old | current_fibrosis_or_scarring_0 | 287 | 163 | 52 | 0.6830 | 0.3121 | 0.2331 | 0.0008 | 0.1119 |
| old | current_fibrosis_or_scarring_1 | 109 | 73 | 23 | 0.7169 | 0.3919 | 0.2971 | 0.0010 | 0.1136 |
| direct_C2 | all_test | 396 | 200 | 75 | 0.6709 | 0.3074 | 0.1691 | 0.0014 | 0.1241 |
| direct_C2 | T0_T1 | 199 | 199 | 30 | 0.7126 | 0.2782 | 0.2053 | 0.0013 | 0.1090 |
| direct_C2 | T1_T2 | 197 | 197 | 45 | 0.6380 | 0.3498 | 0.1241 | 0.0015 | 0.1394 |
| direct_C2 | future_cancer_2y_positive | 75 | 48 | 75 | NA | NA | NA | 0.0016 | 0.1522 |
| direct_C2 | future_cancer_2y_negative | 321 | 170 | 0 | NA | NA | NA | 0.0013 | 0.1176 |
| direct_C2 | future_cancer_within_1y | 31 | 31 | 31 | NA | NA | NA | 0.0020 | 0.1987 |
| direct_C2 | future_cancer_after_1y | 165 | 98 | 44 | 0.5370 | 0.3005 | 0.0242 | 0.0013 | 0.1192 |
| direct_C2 | current_isolation_suspicious_0 | 270 | 154 | 47 | 0.6709 | 0.2635 | 0.1561 | 0.0014 | 0.1214 |
| direct_C2 | current_isolation_suspicious_1 | 126 | 81 | 28 | 0.6866 | 0.4140 | 0.1690 | 0.0014 | 0.1301 |
| direct_C2 | current_nodule_ge_4mm_0 | 272 | 154 | 48 | 0.6605 | 0.2618 | 0.1491 | 0.0014 | 0.1210 |
| direct_C2 | current_nodule_ge_4mm_1 | 124 | 79 | 27 | 0.7068 | 0.4207 | 0.1929 | 0.0014 | 0.1310 |
| direct_C2 | current_emphysema_0 | 251 | 142 | 46 | 0.6775 | 0.3262 | 0.1934 | 0.0014 | 0.1189 |
| direct_C2 | current_emphysema_1 | 145 | 90 | 29 | 0.6525 | 0.3041 | 0.1252 | 0.0014 | 0.1331 |
| direct_C2 | current_fibrosis_or_scarring_0 | 287 | 163 | 52 | 0.6642 | 0.2998 | 0.1481 | 0.0013 | 0.1208 |
| direct_C2 | current_fibrosis_or_scarring_1 | 109 | 73 | 23 | 0.6926 | 0.3846 | 0.2143 | 0.0016 | 0.1329 |

Subgroups include transition type, future cancer/non-cancer, cancer within 1y/later, and validated current suspicion, nodule≥4mm, emphysema, fibrosis/scarring fields. No new clinical label is invented. Tiny or one-class subgroups have undefined discrimination metrics; they still receive geometric and risk-error summaries. These subgroup analyses are descriptive.

analysis/test_pair_errors.csv and test_patient_errors.csv record probability/logit errors, event timing, transitions, true-delta norm and current-future cosine. Best/worst patient tables aggregate eligible transitions within a patient. Rank extremes are descriptive, not a new validation cohort. Error associations are exploratory and not causal; pair-level Spearman p-values do not account for repeated transitions and must not be used as clinical evidence. Plots compare delta norms/cosines, train-fitted true-delta PCA with predicted deltas in the SAME coordinate system, patient-transition pairing, current/predicted/real risk scores, and oracle-gap closure. No UMAP or geometric plot establishes prognosis.

## 11. Limitations

This enriched 999-patient cohort and its test split have already been inspected in earlier research. It is not independent confirmation or representative NLST risk calibration. COLIPRI pretraining included NLST images. The oracle is trained on the same training patients as the predictor; teacher in-sample overfitting is possible, although validation and test patients are never used in gradients. Cross-fitted teacher supervision is untested. Predicted training features are in-sample for subsequent probes; frozen-head evaluation therefore provides an important complementary view.

Clinical-label supervision can learn a stronger current-state classifier rather than recover patient-specific future anatomy. Risk-only and geometry-only controls help assess this, but success through the optimized teacher alone does not prove general future-state quality. Independent probes and geometric errors should be read together; adversarial teacher directions or off-manifold states remain possible. An oracle advantage can reflect irreducible disease/scan uncertainty and does not establish that its entire gap is predictable.

The prospective query is fixed at 365 days, whereas future scans occur at variable actual intervals; the target remains after the actual future landmark for comparability. Fixed query controls prevent future timing leakage but do not eliminate uncertainty from off-schedule scanning. There is no validated significant-change label, new survival model, or additional clinical conditioning. Optional 8³ tokens/K=8/supervision disease-projector ablations are deferred so this remains a global loss experiment. A negative result is about this finite deterministic family, not proof of information-theoretic impossibility.

## 12. Gate decision

| outcome | recommendation | primary_candidate | meaningful_auroc_gain | risk_oracle_gap_reduction_supported | downstream_improvement_supported | geometry_improves | alignment_mae_improves | downstream_equivalence_supported | rule | scope | no_world_model_implemented |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C (tested deterministic family) | No meaningful positive gain recovered by the validation-selected predictor; stop enlarging this deterministic forecasting family. | direct_C2 | 0.0200 | False | False | False | False | False | Primary candidate selected by validation. Gain >=0.02 and paired 95% interval above zero. Equivalence requires interval entirely inside ±0.02. Outcome C requires upper bounds to exclude +0.02; otherwise inconclusive. | Evidence about tested architectures/losses, not a proof that no deterministic mapping can recover future information. | True |

**Final recommendation: No meaningful positive gain recovered by the validation-selected predictor; stop enlarging this deterministic forecasting family.** A positive outcome justifies planning an independent probabilistic follow-up; it does not cause a world model to be built in this gate. Where confidence intervals remain wide, the result stays inconclusive rather than being forced into Outcome B/C. Risk-only control comparisons are included in the paired table; if the candidate does not outperform that control, avoid attributing gains uniquely to future anatomic dynamics.

Reproduce: `PYTHONPATH=src:scripts .venv/bin/python scripts/run_nlst_clinical_predictor_gate.py`. `--smoke` writes a separate verification run; `--analysis-only` regenerates completed analysis without training. Configs, source/input hashes, patient manifests, scalers, validation selections, checkpoints, CSV logs and offline W&B are retained.


## Validation and control attribution

| channel | head | model | comparator | task | metric | gain | ci_low | ci_high | valid_draws | primary_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risk | frozen_oracle | direct_C2 | direct_C1 | primary_future_cancer_2y | auroc | 0.0200 | -0.0006 | 0.0410 | 2000 | True |
| direct | linear | direct_C2 | direct_C1 | primary_future_cancer_2y | auroc | 0.0121 | -0.0053 | 0.0325 | 2000 | True |
| direct | mlp | direct_C2 | direct_C1 | primary_future_cancer_2y | auroc | 0.0186 | 0.0005 | 0.0372 | 2000 | True |
| risk | frozen_oracle | direct_C2 | old | primary_future_cancer_2y | auroc | -0.0217 | -0.0430 | -0.0027 | 2000 | True |
| risk | frozen_oracle | direct_C2 | identity | primary_future_cancer_2y | auroc | -0.0141 | -0.0372 | 0.0086 | 2000 | True |
| risk | frozen_oracle | direct_C2 | current | primary_future_cancer_2y | auroc | -0.0141 | -0.0372 | 0.0086 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_geometry_only | primary_future_cancer_2y | auroc | -0.0224 | -0.0453 | -0.0012 | 2000 | True |
| direct | linear | direct_C2 | residual_geometry_only | primary_future_cancer_2y | auroc | -0.0026 | -0.0249 | 0.0200 | 2000 | True |
| direct | mlp | direct_C2 | residual_geometry_only | primary_future_cancer_2y | auroc | -0.0015 | -0.0127 | 0.0093 | 2000 | True |
| risk | frozen_oracle | direct_C2 | residual_risk_only | primary_future_cancer_2y | auroc | -0.0157 | -0.0405 | 0.0071 | 2000 | True |
| direct | linear | direct_C2 | residual_risk_only | primary_future_cancer_2y | auroc | 0.0021 | -0.0189 | 0.0224 | 2000 | True |
| direct | mlp | direct_C2 | residual_risk_only | primary_future_cancer_2y | auroc | 0.0099 | -0.0018 | 0.0223 | 2000 | True |

Against the capacity-matched state+delta control, supported loss-specific superiority: **False**. Against the same-backbone risk-only control, supported superiority: **False**. Both use the predeclared 0.02 AUROC gain and paired CI above zero; absence of superiority does not demonstrate equivalence. The main gate outcome concerns clinically recoverable information, while these controls constrain attribution to clinical loss or future anatomy.

Risk/error tables use seed-ensemble probabilities. Delta/PCA plots use mean predicted embeddings across seeds. For nonlinear teachers, averaging embeddings and averaging their probabilities are different operations; per-seed alignment metrics remain available.

Every inferred state is a deterministic function of the current embedding and fixed forecast query. Thus [current,predicted future] contains no new information beyond those inputs. A downstream gain indicates improved clinical decodability/inductive bias, not acquisition of an unobserved future measurement. The learned oracle AUROC is a head-dependent reference, not an information-theoretic upper bound. The oracle-gap analysis tests preservation of future-risk semantics rather than proof of patient-specific future anatomy.

The validated report includes unchanged pairs/splits, all prediction identities, exact old checkpoint reproduction, identical backbone counts, masked endpoint sample counts, and 2,000 patient-cluster bootstrap draws.
