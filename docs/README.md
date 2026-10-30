# Documentation guide

The repository supports the LUNG1 development study and NSCLC-Radiogenomics external evaluation described in the [manuscript](../manuscript/main.tex).

## Start here

- [Study synthesis (Italian)](lung1_fm_sintesi_finale.md)
- [Training, preprocessing and results overview](lung1_foundation_models_training_preprocessing_results.md)
- [Cohort audit](lung1_cohort_audit.md)
- [Original research proposal](planning/research_proposal.md)
- [Experiment progress record](planning/experiment_progress.txt)

## Protocols and principal analyses

| Topic | Protocol / setup | Report |
| --- | --- | --- |
| Frozen survival probes | [E1](lung1_e1_frozen_cox.md), [crop and ridge](lung1_e1_crop_and_ridge_protocol.md) | [E1 results](lung1_e1_results_report.md), [crop/ridge](lung1_e1_crop_ridge_results_report.md) |
| Survival adaptation | [E2 setup](lung1_e2_setup_and_run.md), [selected protocol](lung1_bf16_selected_protocol.md) | [Selected baselines](lung1_bf16_selected_baselines_report.md) |
| Data efficiency | [E3/E5 LoRA](lung1_e3_e5_lora_protocol.md) | [Frozen linear](lung1_e3_frozen_linear_report.md), [matched baselines](lung1_e3_matched_baselines_report.md) |
| Tumour dependence | [GTV token mapping](lung1_e4_gtv_token_mapping_report.md) | [Token masking](lung1_e4_token_masking_report.md), [regional Cox](lung1_e4_regional_frozen_cox_report.md) |
| Clinical incremental value | [Phase 1](lung1_phase1_protocol.md) | [Clinical plus FM](lung1_clinical_plus_fm_report.md) |
| Tumour-centred baselines | [Phase 2](lung1_phase2_protocol.md) | [Phase 2 results](lung1_phase2_tumour_centred_report.md) |
| Encoder adaptation sweep | [Phase 3](lung1_phase3_protocol.md), [validation selection](lung1_sweep_validation_protocol.md) | [DINOv3 selected](lung1_dinov3_bf16_selected_report.md) |
| External evaluation | [Phase 4](lung1_phase4_protocol.md) | [External results](lung1_phase4_external_validation_report.md) |

Reports preserve the experiment history, including exploratory and superseded choices. For the manuscript's final interpretation and outstanding checks, consult [verification](../manuscript/verification.md) and the [evidence inventory](../manuscript/evidence_inventory.md). Protocol and report filenames remain stable because scripts and provenance records reference them.
