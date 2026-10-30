# Working-draft verification and open issues

Date: 2026-10-26. This is a bounded implementation and evidence check, not completion of the ARS integrity gate, an independent peer review, or a TRIPOD+AI compliance certificate.

## Completed checks

- pdfLaTeX/BibTeX build succeeded. Final logs contain no undefined citations/references, overfull boxes, or BibTeX warnings.
- Every in-text citation key resolves to an entry in `references.bib`.
- All 24 phase 3 summary files report five completed folds. All reported sweep table values are generated from `mean_fold.uno_c_36m`, not pooled metrics.
- External table values and intervals are generated from `outputs/phase4_external/external_validation.json`; the primary denominator and 36-month event count were checked.
- SHA256 hashes for all 25 numerical source files replay successfully against `artifact_sources.json`.
- Sweep PNG and its rendered manuscript page were visually inspected for readable labels, shared scale, and cropping.
- Rough section-source length is 2,590 whitespace-delimited words including LaTeX and material-gap annotations. This is an initial draft below the eventual 4,000–5,000-word planning target.

## Claim provenance

| Claims | Evidence used | Current status |
|---|---|---|
| Phase 3 scores, complete grid, five folds per setting | `outputs/lung1_phase3/*/*/*/summary.json` | Directly checked; generated table and figure |
| External primary scores, intervals, primary paired contrasts, and mortality calibration | `outputs/phase4_external/external_validation.json` | Directly inspected; score table generated |
| Internal frozen reference values | `internal_lung1_uno_c_36m` fields in the external JSON | Directly read from downstream artifact; original internal-output reconciliation remains |
| Internal COLIPRI vs stage/volume confidence interval | Matched-baseline report and updated synthesis | Report-derived; explicitly flagged in draft |
| Selected-checkpoint scores and 221/111/test split | `docs/lung1_foundation_models_training_preprocessing_results.md` | Report-derived; retain separate pooled estimand |
| PCA8 and PCA16 learning-curve values and chronology | `docs/lung1_e3_frozen_linear_report.md` | Report-derived; post hoc PCA16 identified |
| Token-masking differences and embedding changes | `docs/lung1_e4_token_masking_report.md` | Report-derived; no causal attribution claimed |
| Cohort exclusions and eligibility | `docs/lung1_cohort_audit.md` and external report | Audit/report-derived; no raw patient records copied into manuscript |
| Clinical coding and imputation | Phase 1 protocol and implementation inspection | Legacy T5/N4 uncertainty surfaced, not resolved |

## Required follow-up

1. Complete model-specific and external-dataset bibliographic metadata; identify exact pretrained checkpoints and audit any pretraining/evaluation overlap. The external TCIA page returned HTTP 500 during retrieval; it is not presented as verified.
2. Verify external endpoint time origin and resolve author-facing clinical coding assumptions. TCIA states that T5/N4 legacy meanings are unclear; our grouping is an analytic convention, not a confirmed semantic mapping.
3. Reconcile report-derived claims to original aggregate artifacts and preserve the exact bootstrap families and estimands. No new inferential comparison was calculated during drafting.
4. E5 external LoRA reported in Results (2026-10-27, from `outputs/lung1_e5_lora_v100_fp16/*/summary.json`); adaptation-fixed LoRA/partial values in Table trainable_results except CT-CLIP partial (running). Paired tests vs PCA16 still pending. Preserve test-informed selection and FP16/BF16 distinctions; the current draft does not infer unfinished results.
5. Complete figure set, cohort table, preprocessing supplement, and final abstract. Existing sweep figure is new and reproducible; no claim is made that the full planned figure package is complete.
6. Obtain author declarations: authors/affiliations, contributions, funding, conflicts, ethics and data-use statement, code availability, and AI-use disclosure review.
7. Run full scientific and citation review before any submission or publication-ready claim. The target journal is still open.
