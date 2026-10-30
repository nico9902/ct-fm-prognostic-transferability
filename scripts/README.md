# Script guide

Run scripts from the repository root. Scripts remain in one directory because they import shared analysis functions from one another. Shell launchers encode experiment-specific settings; inspect them before running a GPU workload.

| Stage | Entry points |
| --- | --- |
| LUNG1 cohort | `build_lung1_cohort.py`, `add_lung1_gtv_volume.py` |
| External cohort | `build_radiogenomics_cohort.py`, `preprocess_radiogenomics.py` |
| Preprocessing and caches | `preprocess_lung1_luna_style.py`, `cache_lung1_e2_inputs.py` |
| Frozen features | `extract_lung1_e1_embeddings.py`, `extract_lung1_e2_tokens.py` |
| Shared encoder loading | `foundation_model_loaders.py` (Rad-JEPA and TANGERINE) |
| Frozen Cox and nested selection | `run_lung1_frozen_cox_e1.py`, `run_lung1_e3_frozen_linear.py`, `select_lung1_nested_heads.py`, `analyze_lung1_nested.py` |
| Clinical/radiomic baselines | `extract_lung1_e3_radiomics.py`, `run_lung1_e3_baselines.py`, `run_lung1_ceiling_ladder.py`, `run_lung1_clinical_plus_fm_cox.py` |
| Trainable survival heads | `run_lung1_e2_mlp.py`, `run_lung1_e2_sqa.py`, `run_lung1_bf16_selected_heads.py` |
| Encoder adaptation | `run_lung1_e2_encoder_adaptation.py`, `run_lung1_e2_encoder_bf16_lightning.py`, `run_lung1_e5_lora.py` |
| Tumour-centred models | `extract_lung1_tumor_patch_embeddings.py`, `run_lung1_hosny_finetune.py`, `hosny_deep_prognosis.py` |
| Tumour masking | `extract_lung1_e4_token_masked_embeddings.py`, `run_lung1_e4_token_masking_cox.py`, `run_lung1_e4_no_refit_cox.py` |
| Representation probing | `run_lung1_embedding_probing.py` |
| External evaluation | `freeze_lung1_models.py`, `apply_radiogenomics_external.py`, `analyze_external_lora_calibration.py` |
| Shared Cox optimizer | `ridge_cox_lbfgs.py` |

`launch_*` scripts orchestrate runs; `analyze_*`, `compare_*`, `summarize_*` and statistical `test_*` scripts consume experiment outputs. The latter are research analyses, distinct from the unit tests in `tests/`.

See the [root README](../README.md) for environment setup. Input data and pretrained weights must be supplied separately.
