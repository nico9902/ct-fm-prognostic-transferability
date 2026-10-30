# LUNG1 E3: frozen-representation label-efficiency curves

**Run date:** 9 October 2026  
**Cohort:** 416 NSCLC-Radiomics/LUNG1 patients; 369 observed deaths  
**Scope:** four frozen CT foundation-model global embeddings and the same linear Cox probe at four training-label budgets

## Question and design

This E3 experiment asks how much labeled LUNG1 data is needed to decode overall-survival information from each frozen representation. It does not test survival-query adaptation, LoRA, or fine-tuning. The label budget covers all patients used to fit the Cox model **and** select its penalty.

The original-CT embedding caches and repetition zero of the locked five-fold E1 assignments were reused. Each outer fold has 332–333 development patients and 83–84 untouched test patients. Within each development pool, 10%, 25%, and 50% subsets were drawn ten times. Each draw is nested (10% within 25% within 50% within 100%) and shared across all four models. Sampling preserves event status and early/late stage at 50% and 25%; the 10% step preserves event status because the early-stage censored stratum can contain only one or two patients. The 100% model is fitted once per fold.

For every model, fold, draw, and budget, training-only standardization and a **fixed eight-component PCA** precede an L2-regularized linear Cox fit. The penalty is selected from 0.01, 0.1, and 1.0 using three event-stratified inner folds contained entirely within the budget. The final model is refitted on that budget. No patient from an outer test fold enters fitting or penalty selection. After seeing that the eight-component full-budget scores were below the earlier 32-component E1 scores, we ran a **post hoc, fixed 16-component PCA sensitivity** on the identical patient subsets. The eight-component run remains the primary analysis.

The primary metric is the **mean of five outer test-fold Uno C-indices through 36 months**. Harrell C and Uno C over full follow-up, time-dependent AUC at one, two, and three years, and integrated Brier score (IBS) from three to 36 months are secondary. Uno C, AUC, and IBS use a censoring curve estimated from the full *outer development pool* for evaluation only. This keeps evaluation weights identical across budgets within a fold; those additional outcomes are not used to fit embeddings, PCA, Cox, or select the penalty. Full-follow-up Uno C includes deaths through day 3,653, the latest observed death. Results are fold means, so they must not be compared numerically with the pooled selected-checkpoint table without accounting for aggregation and training-protocol differences.

## Sampled supervision

Counts below are means across folds and draws. The 10% subsets each contain 33 patients, 29 deaths over full follow-up, and four censored observations. Deaths within 36 months vary with the selected patients.

| Budget | Training patients | Observed deaths, full follow-up | Deaths within 36 months | Censored |
| --- | ---: | ---: | ---: | ---: |
| 10% | 33.0 | 29.0 | 22.7 | 4.0 |
| 25% | 83.0 | 74.0 | 57.4 | 9.0 |
| 50% | 166.8 | 148.0 | 114.6 | 18.8 |
| 100% | 332.8 | 295.2 | 228.8 | 37.6 |

## Primary eight-component results

| Frozen encoder | Uno C, 10% | Uno C, 25% | Uno C, 50% | Uno C, 100% | Log-event AULC | Full-follow-up Harrell C, 10% → 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | 0.550 | 0.561 | 0.575 | **0.584** | **0.567** | 0.548 → 0.582 |
| TANGERINE | 0.539 | 0.547 | 0.562 | 0.572 | 0.554 | 0.538 → 0.569 |
| Rad-JEPA-3D | 0.511 | 0.516 | 0.524 | 0.536 | 0.520 | 0.511 → 0.539 |
| CT-CLIP | 0.522 | 0.520 | 0.529 | 0.533 | 0.525 | 0.521 → 0.533 |

The [event-efficiency plot](../outputs/lung1_e3_frozen_linear/event_efficiency_curves.png) shows held-out Uno C against the mean number of observed training deaths. Shaded bands are 95% paired patient-bootstrap intervals conditional on the fitted models, fixed folds, and ten subset draws; they do not include model-refitting uncertainty. Log-event AULC is the trapezoidal area under this curve using log₂ observed deaths, divided by the plotted log₂ event-count range. Across the ten draws, the standard deviation of the five-fold mean Uno C at 10% is 0.018 for COLIPRI, 0.017 for TANGERINE, 0.013 for Rad-JEPA-3D, and 0.014 for CT-CLIP.

The 10%-to-100% Uno C gains are +0.034 for COLIPRI, +0.033 for TANGERINE, +0.025 for Rad-JEPA-3D, and +0.011 for CT-CLIP. Conditional paired 95% intervals for the 10%-minus-100% difference are [−0.059, −0.008], [−0.060, −0.009], [−0.056, 0.000], and [−0.039, 0.017], respectively. All six between-model gain-difference intervals include zero; for example, COLIPRI minus CT-CLIP's gain is +0.024 [−0.015, 0.060]. These are descriptive and have no multiplicity adjustment. The curves do not establish a clear model-by-budget interaction; they mainly show loss of discrimination when fewer labels are available, with substantial overlap in uncertainty.

IBS improves from 10% to 100% for every encoder: COLIPRI 0.237 → 0.211, TANGERINE 0.230 → 0.209, Rad-JEPA-3D 0.235 → 0.215, and CT-CLIP 0.244 → 0.215. The complete AUC, Brier, calibration, full-follow-up concordance, selected penalty, and per-fit event counts are in the output tables.

## Fixed 16-component sensitivity

The [16-component event-efficiency plot](../outputs/lung1_e3_frozen_linear_pca16/event_efficiency_curves.png) uses the same patient subsets, folds, penalty grid, and evaluation rule. Only the PCA dimension changes. Its 620 fits and secondary metrics are saved separately.

| Frozen encoder | Uno C, 10% | Uno C, 25% | Uno C, 50% | Uno C, 100% | Log-event AULC | Full-follow-up Harrell C, 10% → 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | 0.550 | 0.577 | 0.599 | **0.617** | **0.584** | 0.549 → 0.614 |
| TANGERINE | **0.558** | 0.576 | 0.595 | 0.599 | 0.582 | 0.555 → 0.595 |
| Rad-JEPA-3D | 0.515 | 0.525 | 0.526 | 0.539 | 0.525 | 0.515 → 0.540 |
| CT-CLIP | 0.512 | 0.521 | 0.527 | 0.530 | 0.522 | 0.512 → 0.529 |

Increasing PCA capacity changes the learning curves materially. COLIPRI's 100% Uno C rises by 0.033, while its 10% score is essentially unchanged; its 10%-to-100% gain becomes +0.067. TANGERINE improves at both ends, and it slightly exceeds COLIPRI at 10%. At 10%, the larger probe worsens IBS for COLIPRI (0.237 → 0.264) and TANGERINE (0.230 → 0.238), consistent with less stable survival-probability estimates at that budget. Conditional bootstrap intervals for the larger COLIPRI gain versus Rad-JEPA-3D and CT-CLIP exclude zero by small margins, but this PCA setting was examined after the primary results, and the contrasts have no multiplicity adjustment. They are exploratory. The robust conclusion is that the apparent low-label ranking and gain depend on probe capacity.

## Interpretation and limits

This is an internal learning-curve experiment on a cohort already used for E1/E2 development; it is not external validation. It measures the combined effect of training-patient count and event count. Because approximately 89% of LUNG1 patients have an observed death, these counts move together and the experiment cannot isolate the causal effect of event count.

The fixed PCA dimension makes probe capacity comparable across budgets but can discard prognostic information present in later components. For context, the earlier 32-component E1 probe reached a higher full-data COLIPRI score under a different protocol. The 16-component sensitivity demonstrates that capacity affects E3 conclusions. Small 10% subsets also make penalty choice and Breslow baseline estimates variable. The patient-bootstrap intervals are conditional on the fitted models and selected subsets; they do not represent uncertainty from collecting a new LUNG1-like cohort.

The [matched stage + tumor volume and GTV radiomics Cox baselines](lung1_e3_matched_baselines_report.md) use these same patient subsets and outer folds. Their primary Uno C curves are close to the eight-component COLIPRI curve; the report includes paired differences, a plot, and the input-annotation caveat for radiomics.

## Reproduction and artifacts

Run the cached-embedding fits and analysis with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e3_frozen_linear.py
MPLCONFIGDIR=/tmp/lung1_e3_matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_lung1_e3_frozen_linear.py --bootstrap 300
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_lung1_e3_frozen_linear.py --components 16 --output-dir outputs/lung1_e3_frozen_linear_pca16
MPLCONFIGDIR=/tmp/lung1_e3_matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_lung1_e3_frozen_linear.py --output-dir outputs/lung1_e3_frozen_linear_pca16 --bootstrap 300
```

The [E3 output directory](../outputs/lung1_e3_frozen_linear) contains `protocol.json`, locked `subsets.csv`, 620 per-fit JSON files, `fold_metrics.csv`, `draw_metrics.csv`, `summary.csv`, held-out predictions, `paired_uno_bootstrap.json`, the figure, and a `COMPLETE` marker. The 16-component sensitivity has its own output directory and uses identical subset IDs. Patient-level artifacts remain local under `outputs/`.

## DINOv3 addition (17 October 2026)

DINOv3 was added to both E3 output directories with identical subsets and probes; see [the DINOv3 E3/E4 report](lung1_dinov3_e3_e4_report.md). Earlier fits and results are unchanged.
