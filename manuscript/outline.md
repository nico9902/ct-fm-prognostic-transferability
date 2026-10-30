# Detailed outline and argument map

2026-10-26. Elaborates the four-question structure accepted in the conversation; author authorized proceeding after configuration confirmation. This is a writing outline, not a retrospective preregistration.

| Section | Planned words | Purpose and evidence | Boundary |
|---|---:|---|---|
| Introduction | 500–650 | Lesion-centred biomarker precedent; general CT representations; survival transfer question. Pai, Hosny, model sources. | No claim to be the first study without a broader search. |
| Methods | 1,200–1,500 | Audited cohorts; endpoint; model-specific inputs; locked folds; frozen/selected/sweep distinctions; perturbations; external evaluation. | Identify post hoc PCA16 and test-informed extension; no invented ethics statements. |
| Results 1: frozen signal | 350–450 | Internal frozen discrimination and clinical comparisons. | PCA16 is a presentation reference, not the original E3 primary. |
| Results 2: adaptation and budget | 450–600 | All 24 phase 3 configurations; initial selected protocol; PCA8/PCA16 curves. | No significance test or selected-model guarantee for sweep maxima; E3/E5 pending. |
| Results 3: perturbations | 200–300 | Native token masking, matched controls, refit/no-refit, clinical probing. | Robustness does not establish absence of tumour information. |
| Results 4: external transfer | 300–400 | Primary and secondary populations, paired contrasts, calibration. | Conditional uncertainty; no equivalence from nonsignificance. |
| Discussion | 800–1,000 | Transferable information vs incremental utility; optimization dependence; perturbation interpretation; limitations. | Separate evidence from explanations and proposed future tests. |
| Abstract | 250–300 | Finalize after pending experiments are reviewed. | Provisional placeholder in working draft. |

## Argument chains

1. Several frozen pipelines discriminate survival -> information is accessible under the tested readouts -> this does not establish clinical usefulness or a causal advantage of pretraining.
2. Selected adaptation and fixed-epoch sweep differ substantially -> blanket adaptation-failure claim is contradicted -> sample size, checkpoint rule, and optimization change together.
3. Tumour perturbations produce small discrimination changes -> predictions are relatively robust to those interventions -> redundant information and off-distribution masking remain alternative explanations.
4. External discrimination persists for some models while calibration shifts -> risk ranking and absolute-risk transfer are distinct -> incremental clinical utility needs independent assessment.

## Figure and table plan

- Figure 1: study design with protocol chronology and independent/exploratory analyses clearly distinguished (pending).
- Figure 2: entire adaptation sweep (generated from summary JSONs now); frozen/LoRA learning curves added only when complete.
- Figure 3: external discrimination and calibration (planned; table available now).
- Figure 4: tumour perturbation estimates with controls (planned).
- Table 1: audited cohort summary.
- Table 2: frozen internal reference and external results (generated from external evaluation artifact; internal values carried in that artifact).
- Supplementary Table S1: all 24 phase 3 settings (generated).

## Material gaps

Final authorship, institutions, ethics/data-use declaration, funding, conflicts, code-release location, complete checkpoint/pretraining-overlap audit, final experiment status, and full reference coverage. The current draft is intentionally shorter than the eventual target and is not submission-ready.
