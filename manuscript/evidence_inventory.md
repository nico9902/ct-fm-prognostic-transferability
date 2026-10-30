# Evidence inventory

Snapshot: 2026-10-26. Paths are relative to the repository root. This is an intake inventory, not a completed claim verification audit.

| Evidence block | Source | Use and limitation |
|---|---|---|
| Cohort audit | `docs/lung1_cohort_audit.md` | Verify eligibility, counts, endpoints, and exclusions before final methods |
| Overall existing synthesis | `docs/lung1_fm_sintesi_finale.md` | Updated 2026-10-26 with all phase 3 configurations and revised interpretation; earlier non-phase-3 figures still derive from existing reports |
| Encoder protocols and initial adaptation | `docs/lung1_foundation_models_training_preprocessing_results.md` | Separate selected/pooled analyses from the primary frozen protocol |
| Frozen learning curves | `docs/lung1_e3_frozen_linear_report.md`, `docs/lung1_e3_matched_baselines_report.md` | Verify budget matching, repeated draws, and PCA sensitivity |
| Clinical incremental value | `docs/lung1_clinical_plus_fm_report.md`, `docs/lung1_phase1_protocol.md` | Report paired uncertainty; avoid equating nonsignificance with equivalence |
| Tumour-centred comparators | `docs/lung1_phase2_tumour_centred_report.md` | Retain implementation and segmentation-access limitations |
| Regional and perturbation analyses | `docs/lung1_e4_regional_frozen_cox_report.md`, `docs/lung1_e4_token_masking_report.md` | Bound conclusions to the interventions actually evaluated |
| Phase 3 adaptation sweep | `docs/lung1_phase3_protocol.md`, `outputs/lung1_phase3/` | Completed: marker and 24 summaries / 120 fold records checked in this conversation |
| External evaluation | `docs/lung1_phase4_protocol.md`, `docs/lung1_phase4_external_validation_report.md` | Distinguish primary 143-patient and secondary 199-patient populations; verify counts and uncertainty from outputs |
| Exploratory LoRA extension | `docs/lung1_e3_e5_lora_protocol.md` | Configuration chosen after inspecting phase 3; external cohort previously examined |
| Current V100 extension | `outputs/lung1_e3_lora_v100_fp16/`, `logs/lung1_e3_e5_lora_v100/` | Incomplete at the latest check in this conversation; refresh status before using results; FP16 separate from BF16 phase 3 |

## Directly inspected phase 3 values

Metric: `mean_fold.uno_c_36m` from each configuration's `summary.json`. Values below are rounded to four decimals; no new statistical comparison was performed.

| Model | Method | Learning rate | Epochs | Mean-fold Uno C | Source |
|---|---|---|---|---|---|
| COLIPRI | LoRA | 1e-3 | 20 | 0.6393 | `outputs/lung1_phase3/lr1e-3_ep20/lora/colipri/summary.json` |
| TANGERINE | LoRA | 1e-3 | 20 | 0.6301 | `outputs/lung1_phase3/lr1e-3_ep20/lora/tangerine/summary.json` |
| COLIPRI | Partial | 1e-4 | 20 | 0.6005 | `outputs/lung1_phase3/lr1e-4_ep20/partial/colipri/summary.json` |
| TANGERINE | Partial | 1e-4 | 20 | 0.6289 | `outputs/lung1_phase3/lr1e-4_ep20/partial/tangerine/summary.json` |

These rows document the narrative correction, not the final sweep table. The manuscript must include all 24 configurations and must not present test-informed selection as confirmatory optimization.

## Before manuscript claims are finalized

- Reconcile primary and sensitivity protocols, their dates, and analysis populations.
- Trace each numerical claim and confidence interval to the original aggregate output.
- Verify foundation-model, dataset, and comparator references against primary sources.
- Audit cohort overlap with pretraining data where documentation permits; record uncertainty rather than assume independence.
- Verify outcome definitions, censoring estimation, missing-data handling, and calibration estimands.
- Check whether comparisons support superiority, equivalence, or only an inconclusive result.
- Record unresolved material gaps and author-supplied declarations explicitly.

Reporting guidance already located: Collins et al., TRIPOD+AI, BMJ 2024;385:e078378, https://doi.org/10.1136/bmj-2023-078378. A full manuscript bibliography has not yet been assembled.
