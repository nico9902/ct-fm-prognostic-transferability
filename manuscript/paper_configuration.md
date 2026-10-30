# Paper Configuration Record

Date: 2026-10-26
Status: configuration confirmed by the author on 2026-10-26; LaTeX/BibTeX replaces the initially proposed Markdown manuscript format.

| Parameter | Value | Basis |
|---|---|---|
| Working directory | `manuscript/` | Explicit author instruction |
| Topic | Prognostic transferability of CT foundation models in NSCLC | Accepted discussion |
| Research question | How much overall-survival information can be extracted from pretrained CT representations, how does adaptation affect it, and what transfers beyond clinical baselines to an external cohort? | Accepted framing; working formulation |
| Article type | Original empirical research, IMRaD | Author confirmed |
| Discipline | Medical imaging, machine learning, survival prediction | Existing experiments |
| Body and abstract language | English | Author confirmed; conversation remains Italian |
| Working format | LaTeX source (.tex) with BibTeX bibliography (.bib); PDF when compiled | Author requested LaTeX in the confirmation turn |
| Citation style | Vancouver, subject to venue requirements | Author confirmed |
| Target journal | Not selected; no venue alignment claimed | Open |
| Length | Provisional 4,000–5,000 main-text words; adapt to selected venue | Planning assumption, not a venue limit |
| Existing materials | Repository protocols, reports, aggregate outputs, analysis scripts, existing figures | Available locally |
| Literature corpus | Not yet curated or verified for this manuscript | `[MATERIAL GAP]` |
| Authorship, funding, conflicts, ethics/data-use statements | To be supplied or verified | `[MATERIAL GAP]` |
| Style samples | None supplied | Open, optional |
| Review criteria | `criteria_binding_unavailable` | No confirmed venue |

## Accepted scientific organization

Working title: *Prognostic transferability of CT foundation models in non-small-cell lung cancer: adaptation, tumour dependence, incremental clinical value and external evaluation.*

Results organized by four questions:

1. Prognostic information accessible in frozen representations, compared with clinical and tumour-centred baselines.
2. Dependence on adaptation settings and training-label budget.
3. Information associated with representations and sensitivity to tumour perturbations.
4. External discrimination, incremental value over clinical predictors, and calibration.

This records the structure already discussed and accepted, not a newly approved detailed outline.

## Interpretation commitments

- The phase 3 sweep supersedes a blanket conclusion that encoder adaptation fails.
- Sweep maxima are exploratory observations, not unbiased estimates of a selected method's superiority.
- Negative or imprecise comparisons do not establish equivalence or absence of added value.
- Tumour perturbation robustness does not prove that tumour information is absent or unused.
- Clinical-variable probing establishes decodability, not causal attribution of the survival prediction.
- External discrimination and absolute-risk calibration are separate outcomes.
- Preserve distinctions between fold-averaged and pooled metrics, PCA configurations, training cohorts, checkpoint selection, and numerical precision.
- The E3/E5 LoRA extension is exploratory; its configuration followed inspection of phase 3 and the external cohort had already been evaluated.

## Next deliverable

Configuration is confirmed. A first LaTeX draft, detailed outline, targeted literature notes, aggregate-source manifest, and bounded verification report are now available. Next, complete the reference and raw-output audits, remaining figures, and unresolved material gaps. Keep final abstract conclusions provisional until the pending LoRA extension is assessed. New literature and integrity stages will be reported as completed only after they have actually been performed.
