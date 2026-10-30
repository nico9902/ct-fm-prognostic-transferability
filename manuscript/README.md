# NSCLC prognostic transferability manuscript

Working directory established on 2026-10-26.

## Working files

- [Paper configuration](paper_configuration.md): confirmed scope and settings, and remaining decisions.
- [Evidence inventory](evidence_inventory.md): local source documents and interpretation constraints.

The author accepted the proposed scientific framing and four-question results structure in the conversation. The author subsequently confirmed English, IMRaD, and Vancouver citations, with LaTeX/BibTeX as the manuscript format. The target venue remains open. Planning records remain in Markdown. A first working draft with explicit material gaps is available; it is not submission-ready.

## Workflow source

Requested architecture: https://github.com/Imbad0202/academic-research-skills/blob/main/docs/ARCHITECTURE.md

Applied skill: https://github.com/Imbad0202/academic-research-skills/blob/main/academic-paper/SKILL.md (declared version 3.3.1; consulted 2026-10-26).

Intake reference: https://github.com/Imbad0202/academic-research-skills/blob/main/academic-paper/agents/intake_agent.md

These instructions were consulted remotely; the plugin has not been installed. This directory does not claim completion of the repository's automated integrity gates or independent review workflow.

Current stage: first LaTeX working draft compiled; targeted literature search and detailed outline recorded. Full integrity review and reference completion remain outstanding. The configuration checkpoint is satisfied; no further confirmation of the same settings is needed.

## Evidence handling

Use original aggregate outputs to verify numerical statements before drafting final results. Preserve source paths and distinguish report-derived figures from directly checked outputs. Mark missing material explicitly with `[MATERIAL GAP]`. Keep incomplete experiments out of completed-results tables. Do not infer ethics approval, authorship, funding, conflicts, or bibliographic details.

Do not describe the manuscript structure chosen now as a prospectively registered analysis plan. Preserve the actual timing of each experimental decision.

## Read and build

- [PDF working draft](build/main.pdf)
- [LaTeX entry point](main.tex), with separate section files in `sections/`
- [BibTeX references](references.bib), using the installed `vancouver.bst`
- [Detailed outline and argument map](outline.md)
- [Targeted literature search notes](literature_notes.md)
- [Verification and open issues](verification.md)
- [Aggregate-output provenance](artifact_sources.json)

From the repository root, run `make -C manuscript`. This regenerates the aggregate tables and sweep figure, then runs pdfLaTeX, BibTeX, and two further pdfLaTeX passes (a third only if LaTeX requests it). All build products, including the PDF and logs, are written to `build/`; `make -C manuscript clean` removes them. It uses the existing `.venv` for matplotlib. No model training or patient-level processing is performed.

The table and figure generator is `scripts/build_artifacts.py`. Phase 3 values come from all 24 completed summary files; the external table comes from the external-evaluation JSON. Source SHA256 hashes are saved with each regeneration. Other prose estimates are explicitly identified in the verification record as report-derived pending a complete raw-output audit.
