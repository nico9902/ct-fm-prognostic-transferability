# NLST disease-state gate report

**Practical recommendation: IMPROVE PREDICTOR; defer the full probabilistic world model.**

**Predeclared equivalence-based gate verdict: INCONCLUSIVE: do not build the full world model.** No RSSM or probabilistic world model was implemented.

## 1. Dataset and cohort audit

The user approved the available cached cohort: 1,000 selected cancer-balanced patients; 999 embedded patients, 2,984 scans, 995 T0→T1 and 989 T1→T2 pairs. This is a development cohort previously used in multiple experiments, not a new independent test set. It does not satisfy a fresh 2,000-patient or population-representative validation claim. See cohort_audit.md/CSV and longitudinal_table.csv for exact counts and timing distributions.

## 2. Exact targets

All probes predict existing future_cancer_1y/2y/3y of the **t+1 landmark**, using 365/730/1095 days. Diagnosis on or before t+1 is excluded. Known later events or adequate cancer-free follow-up are negatives; inadequate follow-up is missing, never negative. Event timing uses candx_days minus scr_days{round}; censor timing uses canc_free_days minus scr_days{round}. The primary gate endpoint was predeclared as cancer within 2 years after t+1; 1y/3y are secondary. This oracle experiment conditions on an observed future scan and survival without diagnosed cancer to that landmark; it is not a deployable baseline-screening risk estimate. Mortality targets are unavailable. Existing binary endpoints are retained; no new survival task or unvalidated time-dependent metrics were introduced.

## 3. Splits and leakage controls

Existing 599/200/200 train/validation/test patient assignments are preserved. One assignment per patient, source scan split consistency, all cache identities, and predictor pair order are checked explicitly. Both transitions remain in the same split. Split seed 42 is recorded as requested; no new split was generated. All scalers and PCA fit training patients only. Validation masked BCE selects the restored best checkpoint; test is evaluated after fitting and never selects models. Seeds: [42, 123, 2026].

## 4. Frozen COLIPRI representations

Checkpoint SHA256: `97d66586507d1f48577b5149777d02e12d25787435a2a295cac5563df9291c14`; revision `6f2d04e59c8a2ade62f62f0bd376fd757c966983`. Reused official RAS/2mm/HU-clamp/192³ preprocessing and raw mean-pooled 864-dimensional embeddings, matching prior experiments. Full pre-pooling dense features are cached at outputs/colipri_decodability/cache/dense as [864,24,24,24]. The gate uses existing native spatial block means as [64,864] (4×4×4, no PCA) to keep attention small; this retains spatial structure but coarsens the dense grid and can dilute focal lesions. The full tensor is available for a later resolution ablation. It is not called the untouched dense token sequence. Mean-pooling equivalence is verified for every scan. No COLIPRI parameters are trained.

## 5. DiseaseStateProjector

Linear 864→128 token projection; learned 4×4×4 positional embeddings; K=4 learned queries; 4-head cross-attention; residual LayerNorm; 128→256→128 MLP and residual LayerNorm. Output S=[4,128]; query identities are shared across scans; deltas are query-wise. States are flattened (512 dimensions) for probes. Parameters are logged for each training stage in runs/*/done.json and W&B. Supervision: masked equal-horizon prognosis from the individual scan plus 0.25 × masked diagnostic BCE for isolation_suspicious, nodule_ge_4mm, emphysema, fibrosis_or_scarring. All class weights use training scans only. This multi-task objective does not make stable-state similarity dominant. No state predictor is fitted. The same current-scan-trained projector is frozen across D0/D1/D2 for each seed; this removes condition-specific projector learning as a confound. Probe inputs for D0 contain no future scan.

## 6–7. Experiments, metrics, uncertainty

| experiment | head | task | auroc | auprc | accuracy | balanced_accuracy | f1 | mcc | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | linear | primary_future_cancer_1y | 0.697 ± 0.014 | 0.262 ± 0.038 | 0.634 ± 0.014 | 0.634 ± 0.030 | 0.214 ± 0.019 | 0.148 ± 0.034 | 0.634 ± 0.049 | 0.634 ± 0.011 |
| D0 | linear | primary_future_cancer_2y | 0.648 ± 0.011 | 0.363 ± 0.012 | 0.639 ± 0.018 | 0.615 ± 0.031 | 0.377 ± 0.033 | 0.185 ± 0.050 | 0.578 ± 0.054 | 0.653 ± 0.010 |
| D0 | linear | primary_future_cancer_3y | 0.647 ± 0.012 | 0.474 ± 0.011 | 0.612 ± 0.017 | 0.597 ± 0.018 | 0.453 ± 0.020 | 0.178 ± 0.034 | 0.563 ± 0.022 | 0.631 ± 0.015 |
| D0 | mlp | primary_future_cancer_1y | 0.711 ± 0.018 | 0.260 ± 0.019 | 0.666 ± 0.034 | 0.632 ± 0.021 | 0.218 ± 0.018 | 0.149 ± 0.026 | 0.591 ± 0.019 | 0.672 ± 0.037 |
| D0 | mlp | primary_future_cancer_2y | 0.669 ± 0.012 | 0.366 ± 0.030 | 0.662 ± 0.021 | 0.613 ± 0.009 | 0.374 ± 0.011 | 0.186 ± 0.011 | 0.533 ± 0.053 | 0.693 ± 0.038 |
| D0 | mlp | primary_future_cancer_3y | 0.654 ± 0.011 | 0.474 ± 0.008 | 0.621 ± 0.018 | 0.593 ± 0.019 | 0.443 ± 0.022 | 0.172 ± 0.035 | 0.528 ± 0.027 | 0.658 ± 0.019 |
| D1 | linear | primary_future_cancer_1y | 0.849 ± 0.010 | 0.490 ± 0.022 | 0.677 ± 0.032 | 0.746 ± 0.012 | 0.287 ± 0.017 | 0.274 ± 0.018 | 0.828 ± 0.019 | 0.664 ± 0.036 |
| D1 | linear | primary_future_cancer_2y | 0.739 ± 0.009 | 0.521 ± 0.020 | 0.666 ± 0.028 | 0.659 ± 0.011 | 0.424 ± 0.014 | 0.257 ± 0.020 | 0.649 ± 0.034 | 0.670 ± 0.040 |
| D1 | linear | primary_future_cancer_3y | 0.680 ± 0.005 | 0.530 ± 0.004 | 0.638 ± 0.028 | 0.623 ± 0.015 | 0.481 ± 0.013 | 0.226 ± 0.030 | 0.587 ± 0.020 | 0.658 ± 0.046 |
| D1 | mlp | primary_future_cancer_1y | 0.845 ± 0.004 | 0.466 ± 0.007 | 0.799 ± 0.015 | 0.788 ± 0.008 | 0.377 ± 0.017 | 0.360 ± 0.018 | 0.774 ± 0.000 | 0.801 ± 0.016 |
| D1 | mlp | primary_future_cancer_2y | 0.744 ± 0.006 | 0.501 ± 0.024 | 0.708 ± 0.008 | 0.675 ± 0.022 | 0.446 ± 0.024 | 0.290 ± 0.034 | 0.622 ± 0.047 | 0.728 ± 0.008 |
| D1 | mlp | primary_future_cancer_3y | 0.696 ± 0.013 | 0.526 ± 0.016 | 0.656 ± 0.011 | 0.633 ± 0.007 | 0.490 ± 0.012 | 0.247 ± 0.010 | 0.581 ± 0.042 | 0.686 ± 0.031 |
| D2 | linear | primary_future_cancer_1y | 0.844 ± 0.013 | 0.494 ± 0.012 | 0.708 ± 0.030 | 0.763 ± 0.012 | 0.308 ± 0.018 | 0.298 ± 0.019 | 0.828 ± 0.019 | 0.698 ± 0.033 |
| D2 | linear | primary_future_cancer_2y | 0.734 ± 0.005 | 0.503 ± 0.016 | 0.687 ± 0.011 | 0.666 ± 0.008 | 0.433 ± 0.007 | 0.270 ± 0.009 | 0.631 ± 0.038 | 0.700 ± 0.023 |
| D2 | linear | primary_future_cancer_3y | 0.684 ± 0.006 | 0.531 ± 0.005 | 0.652 ± 0.015 | 0.631 ± 0.013 | 0.489 ± 0.014 | 0.242 ± 0.025 | 0.584 ± 0.009 | 0.678 ± 0.019 |
| D2 | mlp | primary_future_cancer_1y | 0.845 ± 0.022 | 0.447 ± 0.073 | 0.777 ± 0.037 | 0.795 ± 0.030 | 0.367 ± 0.045 | 0.358 ± 0.050 | 0.817 ± 0.037 | 0.774 ± 0.039 |
| D2 | mlp | primary_future_cancer_2y | 0.753 ± 0.006 | 0.516 ± 0.024 | 0.714 ± 0.009 | 0.694 ± 0.008 | 0.467 ± 0.007 | 0.319 ± 0.010 | 0.662 ± 0.031 | 0.726 ± 0.017 |
| D2 | mlp | primary_future_cancer_3y | 0.686 ± 0.001 | 0.510 ± 0.023 | 0.646 ± 0.017 | 0.632 ± 0.010 | 0.491 ± 0.010 | 0.242 ± 0.020 | 0.599 ± 0.014 | 0.664 ± 0.028 |
| D_abs | linear | primary_future_cancer_1y | 0.806 ± 0.013 | 0.291 ± 0.060 | 0.652 ± 0.029 | 0.732 ± 0.009 | 0.272 ± 0.013 | 0.255 ± 0.013 | 0.828 ± 0.019 | 0.637 ± 0.033 |
| D_abs | linear | primary_future_cancer_2y | 0.708 ± 0.002 | 0.384 ± 0.031 | 0.651 ± 0.006 | 0.659 ± 0.014 | 0.421 ± 0.013 | 0.252 ± 0.021 | 0.671 ± 0.034 | 0.646 ± 0.010 |
| D_abs | linear | primary_future_cancer_3y | 0.671 ± 0.015 | 0.507 ± 0.032 | 0.634 ± 0.018 | 0.622 ± 0.026 | 0.480 ± 0.031 | 0.222 ± 0.046 | 0.593 ± 0.044 | 0.650 ± 0.007 |
| D_abs | mlp | primary_future_cancer_1y | 0.816 ± 0.006 | 0.283 ± 0.043 | 0.715 ± 0.062 | 0.752 ± 0.025 | 0.309 ± 0.045 | 0.291 ± 0.046 | 0.796 ± 0.019 | 0.708 ± 0.069 |
| D_abs | mlp | primary_future_cancer_2y | 0.723 ± 0.005 | 0.392 ± 0.033 | 0.683 ± 0.036 | 0.677 ± 0.013 | 0.444 ± 0.017 | 0.287 ± 0.025 | 0.667 ± 0.058 | 0.686 ± 0.056 |
| D_abs | mlp | primary_future_cancer_3y | 0.688 ± 0.003 | 0.480 ± 0.011 | 0.657 ± 0.013 | 0.652 ± 0.011 | 0.516 ± 0.015 | 0.279 ± 0.019 | 0.640 ± 0.049 | 0.664 ± 0.034 |
| D_delta | linear | primary_future_cancer_1y | 0.784 ± 0.011 | 0.443 ± 0.024 | 0.756 ± 0.048 | 0.715 ± 0.010 | 0.302 ± 0.029 | 0.263 ± 0.027 | 0.667 ± 0.037 | 0.763 ± 0.055 |
| D_delta | linear | primary_future_cancer_2y | 0.662 ± 0.011 | 0.447 ± 0.002 | 0.706 ± 0.021 | 0.640 ± 0.010 | 0.408 ± 0.013 | 0.239 ± 0.021 | 0.533 ± 0.023 | 0.747 ± 0.030 |
| D_delta | linear | primary_future_cancer_3y | 0.589 ± 0.011 | 0.468 ± 0.016 | 0.636 ± 0.021 | 0.579 ± 0.025 | 0.413 ± 0.032 | 0.152 ± 0.048 | 0.448 ± 0.034 | 0.710 ± 0.016 |
| D_delta | mlp | primary_future_cancer_1y | 0.740 ± 0.045 | 0.425 ± 0.052 | 0.763 ± 0.040 | 0.690 ± 0.033 | 0.288 ± 0.040 | 0.237 ± 0.050 | 0.602 ± 0.037 | 0.777 ± 0.042 |
| D_delta | mlp | primary_future_cancer_2y | 0.647 ± 0.016 | 0.423 ± 0.030 | 0.694 ± 0.008 | 0.619 ± 0.003 | 0.382 ± 0.004 | 0.203 ± 0.005 | 0.498 ± 0.015 | 0.740 ± 0.013 |
| D_delta | mlp | primary_future_cancer_3y | 0.573 ± 0.026 | 0.446 ± 0.029 | 0.645 ± 0.015 | 0.581 ± 0.016 | 0.411 ± 0.021 | 0.158 ± 0.032 | 0.434 ± 0.023 | 0.729 ± 0.016 |
| D_identity | linear | primary_future_cancer_1y | 0.683 ± 0.018 | 0.235 ± 0.036 | 0.598 ± 0.027 | 0.649 ± 0.029 | 0.217 ± 0.019 | 0.162 ± 0.033 | 0.710 ± 0.032 | 0.589 ± 0.026 |
| D_identity | linear | primary_future_cancer_2y | 0.643 ± 0.018 | 0.357 ± 0.005 | 0.642 ± 0.009 | 0.628 ± 0.018 | 0.390 ± 0.019 | 0.205 ± 0.029 | 0.604 ± 0.034 | 0.651 ± 0.003 |
| D_identity | linear | primary_future_cancer_3y | 0.647 ± 0.009 | 0.485 ± 0.011 | 0.624 ± 0.009 | 0.612 ± 0.013 | 0.470 ± 0.015 | 0.204 ± 0.023 | 0.584 ± 0.023 | 0.640 ± 0.007 |
| D_identity | mlp | primary_future_cancer_1y | 0.683 ± 0.019 | 0.252 ± 0.005 | 0.658 ± 0.023 | 0.633 ± 0.010 | 0.216 ± 0.002 | 0.149 ± 0.008 | 0.602 ± 0.049 | 0.663 ± 0.030 |
| D_identity | mlp | primary_future_cancer_2y | 0.656 ± 0.023 | 0.340 ± 0.012 | 0.658 ± 0.036 | 0.612 ± 0.020 | 0.373 ± 0.022 | 0.185 ± 0.034 | 0.538 ± 0.068 | 0.686 ± 0.056 |
| D_identity | mlp | primary_future_cancer_3y | 0.660 ± 0.010 | 0.467 ± 0.031 | 0.626 ± 0.024 | 0.608 ± 0.016 | 0.464 ± 0.015 | 0.199 ± 0.030 | 0.566 ± 0.023 | 0.650 ± 0.038 |
| G0 | linear | primary_future_cancer_1y | 0.733 ± 0.010 | 0.189 ± 0.011 | 0.604 ± 0.011 | 0.652 ± 0.006 | 0.219 ± 0.005 | 0.165 ± 0.007 | 0.710 ± 0.000 | 0.595 ± 0.012 |
| G0 | linear | primary_future_cancer_2y | 0.669 ± 0.023 | 0.311 ± 0.018 | 0.607 ± 0.011 | 0.613 ± 0.021 | 0.375 ± 0.021 | 0.178 ± 0.033 | 0.622 ± 0.038 | 0.603 ± 0.005 |
| G0 | linear | primary_future_cancer_3y | 0.645 ± 0.019 | 0.414 ± 0.010 | 0.620 ± 0.023 | 0.619 ± 0.028 | 0.480 ± 0.031 | 0.216 ± 0.051 | 0.617 ± 0.045 | 0.621 ± 0.019 |
| G0 | mlp | primary_future_cancer_1y | 0.707 ± 0.013 | 0.181 ± 0.004 | 0.652 ± 0.023 | 0.659 ± 0.010 | 0.231 ± 0.010 | 0.177 ± 0.013 | 0.667 ± 0.019 | 0.651 ± 0.025 |
| G0 | mlp | primary_future_cancer_2y | 0.650 ± 0.005 | 0.299 ± 0.010 | 0.621 ± 0.013 | 0.610 ± 0.003 | 0.372 ± 0.003 | 0.175 ± 0.005 | 0.591 ± 0.020 | 0.628 ± 0.021 |
| G0 | mlp | primary_future_cancer_3y | 0.645 ± 0.006 | 0.423 ± 0.004 | 0.630 ± 0.006 | 0.621 ± 0.008 | 0.481 ± 0.010 | 0.221 ± 0.015 | 0.602 ± 0.018 | 0.641 ± 0.009 |
| G1 | linear | primary_future_cancer_1y | 0.878 ± 0.008 | 0.543 ± 0.072 | 0.699 ± 0.015 | 0.773 ± 0.014 | 0.310 ± 0.014 | 0.306 ± 0.018 | 0.860 ± 0.019 | 0.686 ± 0.016 |
| G1 | linear | primary_future_cancer_2y | 0.734 ± 0.004 | 0.469 ± 0.033 | 0.659 ± 0.008 | 0.652 ± 0.025 | 0.415 ± 0.025 | 0.243 ± 0.039 | 0.640 ± 0.061 | 0.664 ± 0.014 |
| G1 | linear | primary_future_cancer_3y | 0.680 ± 0.012 | 0.491 ± 0.008 | 0.646 ± 0.013 | 0.632 ± 0.018 | 0.491 ± 0.023 | 0.243 ± 0.033 | 0.599 ± 0.045 | 0.665 ± 0.022 |
| G1 | mlp | primary_future_cancer_1y | 0.867 ± 0.026 | 0.547 ± 0.065 | 0.766 ± 0.023 | 0.785 ± 0.026 | 0.351 ± 0.029 | 0.340 ± 0.036 | 0.806 ± 0.032 | 0.763 ± 0.023 |
| G1 | mlp | primary_future_cancer_2y | 0.735 ± 0.013 | 0.443 ± 0.032 | 0.706 ± 0.005 | 0.676 ± 0.011 | 0.447 ± 0.013 | 0.290 ± 0.018 | 0.627 ± 0.023 | 0.725 ± 0.005 |
| G1 | mlp | primary_future_cancer_3y | 0.689 ± 0.002 | 0.495 ± 0.014 | 0.646 ± 0.009 | 0.634 ± 0.005 | 0.494 ± 0.006 | 0.246 ± 0.009 | 0.605 ± 0.020 | 0.663 ± 0.019 |
| G2 | linear | primary_future_cancer_1y | 0.883 ± 0.007 | 0.606 ± 0.031 | 0.673 ± 0.020 | 0.754 ± 0.006 | 0.290 ± 0.010 | 0.281 ± 0.009 | 0.849 ± 0.019 | 0.658 ± 0.023 |
| G2 | linear | primary_future_cancer_2y | 0.739 ± 0.010 | 0.487 ± 0.002 | 0.659 ± 0.009 | 0.655 ± 0.003 | 0.419 ± 0.004 | 0.249 ± 0.006 | 0.649 ± 0.008 | 0.661 ± 0.013 |
| G2 | linear | primary_future_cancer_3y | 0.671 ± 0.028 | 0.497 ± 0.018 | 0.641 ± 0.030 | 0.617 ± 0.031 | 0.471 ± 0.035 | 0.216 ± 0.058 | 0.560 ± 0.031 | 0.673 ± 0.030 |
| G2 | mlp | primary_future_cancer_1y | 0.869 ± 0.024 | 0.589 ± 0.003 | 0.761 ± 0.015 | 0.801 ± 0.025 | 0.358 ± 0.024 | 0.356 ± 0.032 | 0.849 ± 0.037 | 0.753 ± 0.014 |
| G2 | mlp | primary_future_cancer_2y | 0.726 ± 0.010 | 0.489 ± 0.015 | 0.668 ± 0.023 | 0.647 ± 0.010 | 0.412 ± 0.012 | 0.238 ± 0.019 | 0.613 ± 0.013 | 0.680 ± 0.032 |
| G2 | mlp | primary_future_cancer_3y | 0.666 ± 0.007 | 0.497 ± 0.007 | 0.641 ± 0.018 | 0.615 ± 0.008 | 0.469 ± 0.007 | 0.214 ± 0.018 | 0.555 ± 0.018 | 0.676 ± 0.032 |
| G3 | linear | primary_future_cancer_1y | 0.722 ± 0.013 | 0.188 ± 0.022 | 0.597 ± 0.027 | 0.663 ± 0.037 | 0.224 ± 0.023 | 0.177 ± 0.041 | 0.742 ± 0.065 | 0.584 ± 0.027 |
| G3 | linear | primary_future_cancer_2y | 0.663 ± 0.008 | 0.315 ± 0.007 | 0.620 ± 0.016 | 0.623 ± 0.020 | 0.385 ± 0.020 | 0.195 ± 0.033 | 0.627 ± 0.027 | 0.619 ± 0.014 |
| G3 | linear | primary_future_cancer_3y | 0.661 ± 0.008 | 0.433 ± 0.013 | 0.631 ± 0.007 | 0.630 ± 0.008 | 0.493 ± 0.010 | 0.237 ± 0.015 | 0.628 ± 0.015 | 0.633 ± 0.007 |
| G3 | mlp | primary_future_cancer_1y | 0.698 ± 0.027 | 0.191 ± 0.018 | 0.653 ± 0.044 | 0.645 ± 0.032 | 0.224 ± 0.027 | 0.162 ± 0.039 | 0.634 ± 0.019 | 0.655 ± 0.047 |
| G3 | mlp | primary_future_cancer_2y | 0.644 ± 0.012 | 0.305 ± 0.018 | 0.654 ± 0.004 | 0.620 ± 0.006 | 0.382 ± 0.007 | 0.194 ± 0.009 | 0.564 ± 0.020 | 0.675 ± 0.010 |
| G3 | mlp | primary_future_cancer_3y | 0.660 ± 0.005 | 0.438 ± 0.007 | 0.646 ± 0.003 | 0.635 ± 0.005 | 0.495 ± 0.006 | 0.247 ± 0.008 | 0.608 ± 0.010 | 0.662 ± 0.002 |
| G_abs | linear | primary_future_cancer_1y | 0.847 ± 0.021 | 0.423 ± 0.050 | 0.659 ± 0.015 | 0.751 ± 0.031 | 0.283 ± 0.021 | 0.276 ± 0.035 | 0.860 ± 0.049 | 0.642 ± 0.012 |
| G_abs | linear | primary_future_cancer_2y | 0.713 ± 0.016 | 0.401 ± 0.013 | 0.646 ± 0.013 | 0.656 ± 0.019 | 0.418 ± 0.019 | 0.248 ± 0.030 | 0.671 ± 0.028 | 0.641 ± 0.010 |
| G_abs | linear | primary_future_cancer_3y | 0.660 ± 0.018 | 0.470 ± 0.013 | 0.617 ± 0.019 | 0.603 ± 0.023 | 0.459 ± 0.027 | 0.188 ± 0.042 | 0.569 ± 0.040 | 0.636 ± 0.021 |
| G_abs | mlp | primary_future_cancer_1y | 0.822 ± 0.012 | 0.442 ± 0.041 | 0.726 ± 0.014 | 0.738 ± 0.014 | 0.301 ± 0.013 | 0.276 ± 0.018 | 0.753 ± 0.019 | 0.723 ± 0.014 |
| G_abs | mlp | primary_future_cancer_2y | 0.698 ± 0.016 | 0.381 ± 0.022 | 0.660 ± 0.016 | 0.635 ± 0.016 | 0.399 ± 0.018 | 0.219 ± 0.027 | 0.596 ± 0.028 | 0.675 ± 0.019 |
| G_abs | mlp | primary_future_cancer_3y | 0.674 ± 0.005 | 0.485 ± 0.013 | 0.628 ± 0.011 | 0.614 ± 0.016 | 0.471 ± 0.020 | 0.209 ± 0.029 | 0.581 ± 0.036 | 0.647 ± 0.016 |
| G_delta | linear | primary_future_cancer_1y | 0.811 ± 0.011 | 0.582 ± 0.011 | 0.659 ± 0.013 | 0.717 ± 0.021 | 0.265 ± 0.014 | 0.239 ± 0.023 | 0.785 ± 0.037 | 0.648 ± 0.014 |
| G_delta | linear | primary_future_cancer_2y | 0.671 ± 0.020 | 0.467 ± 0.031 | 0.650 ± 0.034 | 0.632 ± 0.026 | 0.396 ± 0.029 | 0.213 ± 0.045 | 0.604 ± 0.015 | 0.660 ± 0.039 |
| G_delta | linear | primary_future_cancer_3y | 0.592 ± 0.005 | 0.470 ± 0.011 | 0.596 ± 0.022 | 0.565 ± 0.018 | 0.411 ± 0.019 | 0.120 ± 0.035 | 0.493 ± 0.014 | 0.637 ± 0.027 |
| G_delta | mlp | primary_future_cancer_1y | 0.805 ± 0.006 | 0.548 ± 0.004 | 0.792 ± 0.044 | 0.774 ± 0.018 | 0.365 ± 0.039 | 0.344 ± 0.038 | 0.753 ± 0.037 | 0.795 ± 0.050 |
| G_delta | mlp | primary_future_cancer_2y | 0.646 ± 0.017 | 0.455 ± 0.010 | 0.672 ± 0.017 | 0.626 ± 0.011 | 0.388 ± 0.012 | 0.207 ± 0.014 | 0.551 ± 0.056 | 0.700 ± 0.034 |
| G_delta | mlp | primary_future_cancer_3y | 0.578 ± 0.005 | 0.441 ± 0.007 | 0.586 ± 0.013 | 0.548 ± 0.006 | 0.388 ± 0.008 | 0.089 ± 0.011 | 0.460 ± 0.023 | 0.636 ± 0.027 |
| G_identity | linear | primary_future_cancer_1y | 0.714 ± 0.013 | 0.168 ± 0.017 | 0.593 ± 0.023 | 0.646 ± 0.031 | 0.215 ± 0.019 | 0.158 ± 0.034 | 0.710 ± 0.056 | 0.583 ± 0.025 |
| G_identity | linear | primary_future_cancer_2y | 0.660 ± 0.012 | 0.302 ± 0.006 | 0.612 ± 0.017 | 0.621 ± 0.022 | 0.383 ± 0.022 | 0.191 ± 0.035 | 0.636 ± 0.034 | 0.606 ± 0.015 |
| G_identity | linear | primary_future_cancer_3y | 0.653 ± 0.010 | 0.426 ± 0.015 | 0.626 ± 0.004 | 0.623 ± 0.007 | 0.485 ± 0.008 | 0.224 ± 0.013 | 0.617 ± 0.014 | 0.630 ± 0.002 |
| G_identity | mlp | primary_future_cancer_1y | 0.691 ± 0.030 | 0.183 ± 0.023 | 0.649 ± 0.055 | 0.642 ± 0.038 | 0.223 ± 0.031 | 0.160 ± 0.046 | 0.634 ± 0.019 | 0.650 ± 0.058 |
| G_identity | mlp | primary_future_cancer_2y | 0.635 ± 0.016 | 0.299 ± 0.018 | 0.645 ± 0.012 | 0.600 ± 0.015 | 0.361 ± 0.016 | 0.163 ± 0.025 | 0.529 ± 0.020 | 0.672 ± 0.012 |
| G_identity | mlp | primary_future_cancer_3y | 0.651 ± 0.007 | 0.431 ± 0.003 | 0.643 ± 0.005 | 0.631 ± 0.009 | 0.490 ± 0.011 | 0.240 ± 0.016 | 0.602 ± 0.018 | 0.660 ± 0.002 |

Heads: single linear logistic layer or MLP with one 128-unit GELU hidden layer and dropout 0.1. Current-only and delta-only inputs are zero-padded to the concatenated width so within each representation space the architectures and parameter counts match. Global and disease inputs differ in width by design; head family and hidden width remain fixed. Train/val/test metrics, sample counts, and parameter counts are in per_seed_metrics.csv. AUROC, AUPRC and MCC are emphasized. Accuracy and threshold measures use 0.5 fixed in advance; weighted BCE probabilities are not calibrated absolute risks.

Mean ± SD describes 3 independent training seeds. Bootstrap uses 2000 paired patient cluster draws of held-out seed-ensemble predictions and preserves repeated pairs. Its intervals are conditional on fitted models and do not replace training-seed variability. Paired differences and disease-minus-global improvement intervals are in paired_comparisons.csv. No multiplicity adjustment is applied: auxiliary endpoints/ablations are exploratory. Lack of statistical significance is not evidence of equivalence.

## 8. Mandatory no-change control

| space | split | predictor | mse | cosine | l2 | delta_cosine | delta_mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | test | identity | 0.0010 | 0.9753 | 0.8563 | 0.0000 | 0.0010 |
| global | test | learned_seed0 | 0.0009 | 0.9770 | 0.8296 | 0.2795 | 0.0009 |
| global | test | learned_seed1 | 0.0009 | 0.9769 | 0.8315 | 0.2745 | 0.0009 |
| global | test | learned_seed2 | 0.0009 | 0.9768 | 0.8316 | 0.2774 | 0.0009 |
| global | test | learned_seed3 | 0.0009 | 0.9769 | 0.8309 | 0.2792 | 0.0009 |
| global | test | learned_seed4 | 0.0009 | 0.9770 | 0.8281 | 0.2687 | 0.0009 |
| global | test | learned_ensemble | 0.0009 | 0.9772 | 0.8247 | 0.2876 | 0.0009 |
| disease | test | identity_projector_seed42 | 0.4238 | 0.7860 | 12.4570 | 0.0000 | 0.4238 |
| disease | test | identity_projector_seed123 | 0.4245 | 0.7857 | 12.2997 | 0.0000 | 0.4245 |
| disease | test | identity_projector_seed2026 | 0.4470 | 0.7729 | 12.8225 | 0.0000 | 0.4470 |

G3 reuses the existing validation-selected objective A standardized-delta residual MLP, fixed ensemble of prior predictor seeds 0–4. It takes current embedding and interscan interval; future scan, event metadata, and diagnostics are not predictor inputs. Pair indices are verified against its saved manifest. Disease identity metrics are reported separately for each projector seed; there is no learned disease dynamics predictor in this gate. High future-state cosine alone does not establish clinically meaningful forecasting. Learned global forecasts must beat identity reconstruction **and** show downstream prognostic benefit.

## 9. Representation analysis

representation_analysis contains per-pair cosine/L2, cancer-stratified summaries, train-fitted PCA coordinates, and states for all three seeds. plots contains held-out global/disease trajectories connecting scans of each patient and change distributions/delta PCA colored by post-future cancer. Scan trajectories use ever-cancer coloring, explicitly distinct from the landmark-safe pair target. PCA plots are qualitative; geometry does not prove domination by stable anatomy. Significant-change stratification is unavailable.

## 10. Ablations and unavailable tasks

Raw future versus signed delta, delta alone, absolute delta, identity future, linear versus MLP, global versus disease state, and three projector seeds are evaluated. Lesion comparison fields are saved without inventing labels; their codes lack a validated local scan-level significant-change mapping. Progression/change classification and change BCE are unavailable. Diagnostic supervision is not substituted for radiological progression supervision. No pseudo-labels are created. Dense-resolution, K=8, and prognosis-only projector ablations remain untested, so spatial attention versus extra supervision cannot be isolated causally.

## 11–12. Interpretation and gate recommendation

| comparison | head | task | metric | gain | ci_low | ci_high | valid_draws |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 - G0 | linear | primary_future_cancer_2y | auroc | 0.0682 | 0.0237 | 0.1141 | 2000 |
| G2 - G0 | linear | primary_future_cancer_2y | auroc | 0.0703 | 0.0171 | 0.1271 | 2000 |
| G3 - G0 | linear | primary_future_cancer_2y | auroc | -0.0049 | -0.0252 | 0.0145 | 2000 |
| D0 - G0 | linear | primary_future_cancer_2y | auroc | -0.0192 | -0.0647 | 0.0264 | 2000 |
| D1 - D0 | linear | primary_future_cancer_2y | auroc | 0.0886 | 0.0395 | 0.1397 | 2000 |
| D2 - D0 | linear | primary_future_cancer_2y | auroc | 0.0834 | 0.0311 | 0.1376 | 2000 |
| G1 - G3 | linear | primary_future_cancer_2y | auroc | 0.0731 | 0.0290 | 0.1188 | 2000 |
| G2 - G1 | linear | primary_future_cancer_2y | auroc | 0.0021 | -0.0276 | 0.0338 | 2000 |
| (D1-D0) - (G1-G0) | linear | primary_future_cancer_2y | auroc | 0.0204 | -0.0231 | 0.0685 | 2000 |
| (D2-D0) - (G2-G0) | linear | primary_future_cancer_2y | auroc | 0.0131 | -0.0227 | 0.0505 | 2000 |
| G1 - G0 | mlp | primary_future_cancer_2y | auroc | 0.0880 | 0.0484 | 0.1281 | 2000 |
| G2 - G0 | mlp | primary_future_cancer_2y | auroc | 0.0799 | 0.0191 | 0.1434 | 2000 |
| G3 - G0 | mlp | primary_future_cancer_2y | auroc | -0.0017 | -0.0244 | 0.0197 | 2000 |
| D0 - G0 | mlp | primary_future_cancer_2y | auroc | 0.0259 | -0.0133 | 0.0706 | 2000 |
| D1 - D0 | mlp | primary_future_cancer_2y | auroc | 0.0717 | 0.0240 | 0.1217 | 2000 |
| D2 - D0 | mlp | primary_future_cancer_2y | auroc | 0.0836 | 0.0356 | 0.1333 | 2000 |
| G1 - G3 | mlp | primary_future_cancer_2y | auroc | 0.0897 | 0.0429 | 0.1368 | 2000 |
| G2 - G1 | mlp | primary_future_cancer_2y | auroc | -0.0081 | -0.0479 | 0.0338 | 2000 |
| (D1-D0) - (G1-G0) | mlp | primary_future_cancer_2y | auroc | -0.0163 | -0.0516 | 0.0190 | 2000 |
| (D2-D0) - (G2-G0) | mlp | primary_future_cancer_2y | auroc | 0.0037 | -0.0327 | 0.0404 | 2000 |

| head | G1_vs_G0 | G2_vs_G0 | G3_vs_G0 | D0_vs_G0 | D1_vs_D0 | D2_vs_D0 | larger_disease_gain | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | improves | improves | inconclusive | inconclusive | improves | improves | False | INCONCLUSIVE: do not build the full world model |
| mlp | improves | improves | inconclusive | inconclusive | improves | improves | False | INCONCLUSIVE: do not build the full world model |

A meaningful AUROC advantage requires gain ≥0.02 and a paired 95% interval above zero; equivalence requires the entire interval inside ±0.02. These criteria were recorded before training. Discordant heads or intervals covering meaningful gains are inconclusive. Scenario D requires stronger evidence than D0 alone; because validated change labels and independent confirmation are absent, an apparent D candidate warrants replication before building a probabilistic progression model.

**Information equivalence:** [z_t,z_t+1] and [z_t,z_t+1−z_t] are related by an invertible linear transformation. The same holds for disease queries. If G2 outperforms G1, it indicates regularization/conditioning/optimization or finite-sample effects, not stronger information in the change. High similarity is also compatible with sparse meaningful changes. Neither failed probes nor PCA by themselves establish that stable anatomy is the cause.

**Predeclared gate verdict: INCONCLUSIVE: do not build the full world model.** Interpret all evidence within this enriched, previously inspected cohort; COLIPRI vision pretraining itself used NLST images.

Reproduce with `PYTHONPATH=src:scripts .venv/bin/python scripts/run_nlst_gate.py --config results/nlst_gate/configs/gate.yaml`. W&B is logged offline for this run; CSV histories, checkpoints, scalers, input hashes, environment versions, and predictions are saved locally. No external service is necessary to reproduce the report.


## Measured primary comparisons and scientific limits

The projector alone has 251,904 trainable parameters; the frozen image encoder checkpoint has 147,113,088 tensor elements.

**Practical recommendation: IMPROVE PREDICTOR; defer the full probabilistic world model.** The predeclared two-sided equivalence rule remains inconclusive: possible G3 deterioration beyond -0.02 is not ruled out. Lack of meaningful positive gain does not prove exact equivalence.

These paired differences use **seed-ensemble AUROC**,  distinct from the mean ± SD across individual seeds above.

**A: true global future:** linear: +0.068 [95% CI +0.024, +0.114]; mlp: +0.088 [95% CI +0.048, +0.128].

**B: global change:** linear: +0.070 [95% CI +0.017, +0.127]; mlp: +0.080 [95% CI +0.019, +0.143].

**C: current disease versus global:** linear: -0.019 [95% CI -0.065, +0.026]; mlp: +0.026 [95% CI -0.013, +0.071].

**D: excess disease oracle gain:** linear: +0.020 [95% CI -0.023, +0.068]; mlp: -0.016 [95% CI -0.052, +0.019].

**D: excess disease delta gain:** linear: +0.013 [95% CI -0.023, +0.050]; mlp: +0.004 [95% CI -0.033, +0.040].

**Predicted future versus current:** linear: -0.005 [95% CI -0.025, +0.015]; mlp: -0.002 [95% CI -0.024, +0.020].

**True versus predicted future:** linear: +0.073 [95% CI +0.029, +0.119]; mlp: +0.090 [95% CI +0.043, +0.137].

The global predictor ensemble changes test MSE from 0.000965 (identity) to 0.000876, a 9.2% relative reduction. Its test cosine is 0.9772, versus 0.9753 for identity. Reconstruction and clinical discrimination are separate requirements.

An oracle advantage establishes that actual future images contain decodable prognostic information. It does **not** establish that all of that information can be predicted from the current scan. Stochastic disease evolution, scan variation, and events between scans can leave an irreducible oracle gap. Improving a predictor is a research recommendation, not a guarantee that a larger dynamics model will close this gap.

A higher disease-state score, if present, can reflect spatial readout, additional supervision, nonlinear capacity, or regularization. This first gate does not isolate those causes. Greater disease longitudinal gain can also arise from a weaker current-state baseline; both absolute oracle performance and paired improvements should be examined. Without validated radiological-change labels, sensitivity to real radiological progression remains untested.

No result here proves that stable anatomy dominates the global state. The paired oracle test addresses whether clinically useful future information is present; high consecutive cosine alone cannot establish anatomical domination or forecasting success.
