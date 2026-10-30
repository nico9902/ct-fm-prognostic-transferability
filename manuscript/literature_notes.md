# Targeted literature verification

Search date: 2026-10-26. Targeted narrative search, not a systematic review. No exhaustive novelty claim is supported.

Queries covered: CT foundation models COLIPRI/TANGERINE/Rad-JEPA/CT-CLIP; cancer imaging biomarkers Pai 2024; Hosny lung cancer prognosis; Uno censored concordance; TCIA cohort records; LoRA; TRIPOD+AI. Primary publication pages, author repositories/model cards, and PubMed bibliographic records were preferred. Excluded unrelated JEPA applications and secondary summaries as support for model descriptions.

| Source | Verified identifier or URL | Intended role | Verification limit |
|---|---|---|---|
| Pai et al., 2024, Foundation model for cancer imaging biomarkers | 10.1038/s42256-024-00807-9 | Lesion-centred pretraining and low-label prognosis precedent | Publisher text inspected; different task/training protocols preclude direct numerical comparison |
| Hosny et al., 2018, Deep learning for lung cancer prognostication | https://pmc.ncbi.nlm.nih.gov/articles/PMC6269088/ | Supervised tumour-centred precedent | Primary article located; our implementation fidelity is separately limited |
| Uno et al., 2011 | 10.1002/sim.4154 | Censored-survival discrimination | Bibliographic record and author-manuscript excerpt verified |
| Hu et al., 2021, LoRA | https://arxiv.org/abs/2106.09685 | Low-rank adaptation method | Original abstract verified; not evidence of CT survival performance |
| Collins et al., 2024, TRIPOD+AI | 10.1136/bmj-2023-078378 | Reporting guidance | Guidance checked; checklist completion not claimed |
| TCIA NSCLC-Radiomics | https://www.cancerimagingarchive.net/collection/nsclc-radiomics/ | Source cohort and endpoint | Collection page inspected; ambiguous legacy TNM codes noted |
| COLIPRI author model card | https://huggingface.co/microsoft/colipri | Model identity | Model card checked; full paper/checkpoint identity still to reconcile |
| TANGERINE | https://pubmed.ncbi.nlm.nih.gov/41639429/ | Model source | Title, DOI 10.1038/s43856-025-01328-1, journal/year checked; final author metadata pending |
| CT-CLIP | https://www.nature.com/articles/s41551-025-01599-y | Model source | Publisher text checked; checkpoint-mirror limitation remains |
| Rad-JEPA 3D | https://arxiv.org/abs/2607.26196 | Model source | Primary abstract checked; match exact local checkpoint to publication before final citation |
| DINOv3 | https://arxiv.org/abs/2508.10104 | Nonmedical representation comparator | Primary record located; complete bibliographic metadata pending |
| TCIA NSCLC-Radiogenomics | https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/ | External dataset citation | Data citation (Bakr et al. 2017, v4, DOI 10.7937/K9/TCIA.2017.7hs46erv) and descriptor (Bakr et al. 2018, Sci Data 5:180202, DOI 10.1038/sdata.2018.202) verified on the collection page, 2026-10-26 |

The initial BibTeX file contains verified core references only. Model-specific and external-dataset citations remain explicitly marked as material gaps in the draft rather than assigned guessed author/year metadata.

## Model-specific literature context and pretraining-overlap audit (2026-10-26)

Targeted search of primary sources: arXiv abs/html pages, publisher pages, PMC, model cards, author repositories and supplementary files. It is not a systematic review. PubMed, RSNA and some Springer pages blocked automated retrieval. Neukirch 2026 and Allgöwer 2026 were re-checked at abstract level by the main session; the remaining details come from subagent full-text reading and still need a final author check.

### Closest prior work (novelty positioning)

| Source | Cohorts / models | Endpoint / readout | Overlap with this study | Differences |
|---|---|---|---|---|
| Neukirch, Maurer, Strodthoff 2026, arXiv:2607.01001 | LUNG1 dev (338 train / 84 test); external "LUNG2" n=211 (size matches NSCLC-Radiogenomics; the abstract does not name it); Curia, Curia-2, DINOv3, 2D/3D radiomics | Binary 2-y OS AUC (plus volume, stage, histology, age); frozen features + 7 tabular heads; tumour-mask / lung-mask / full-CT inputs | Same cohorts, DINOv3, whole-volume vs tumour-region inputs, external test | No COLIPRI/TANGERINE/Rad-JEPA/CT-CLIP; no Cox/Uno C, calibration, label budgets, LoRA/partial tuning or token masking |
| Aerts, Pai, Hadzic, Fedorov, Mak 2025, TumorImagingBench, Research Square doi:10.21203/rs.3.rs-6630446/v1 (PMC12154149) | NSCLC-Radiomics n=421, NSCLC-Radiogenomics n=133 + 4 others; FMCIB, CT-FM, CT-CLIP, Merlin, VISTA3D, VoCo, SuPreM, PASTA, MedImageInsight, Models Genesis | 2-y OS AUC, kNN on frozen lesion-centred embeddings; occlusion saliency | Multi-FM prognostic benchmark on both cohorts incl. CT-CLIP; tumour-reliance analysis | Lesion-seeded inputs; no whole-volume, adaptation, Cox, or calibration |
| Pai et al. 2024 (pai2024) | Pretrained on DeepLesion; 2-y OS model trained on HarvardRT, tested on LUNG1 (n=420) and "RADIO (NSCLC-Radiogenomics)" (n=133) | Logistic regression on frozen features, AUC | Same cohorts as independent test sets | Lesion patch; binary endpoint |
| Allgöwer et al. 2026, arXiv:2607.08503 | Single-centre lung cancer cohort n=242; CT-CLIP | Survival head; frozen vs LoRA vs full fine-tuning (frozen best per full-text read) | Whole-volume CT FM + LoRA/fine-tuning for lung cancer survival | One FM, no public cohort, no external validation/calibration/masking |
| Hosny et al. 2018 (hosny2018) | Train HarvardRT, tune Radboud, test Maastro (LUNG1 subset n=307) | 2-y OS; retraining with peritumoral region masked lowered AUC (0.70→0.63) | Prior tumour/context ablation on LUNG1 | Retraining-based, not test-time token masking |
| Upadhaya et al. 2024, BJR Open doi:10.1093/bjro/tzae038 | RTOG 0617, FMCIB-type features | 2-y OS | Lung FM prognosis | Different cohort |
| Guetarni et al. 2026, arXiv:2608.00071 | Head and neck: CT-FM, CT-CLIP, VISTA3D, SuPreM | Recurrence | Same paradigm, other cancer | — |
| Chevli et al. 2026, arXiv:2608.05960 | 10 3D CT encoders, frozen features | Thoracic findings | Small focal lesions poorly captured by globally pooled embeddings (motivation for perturbation analysis) | Not prognosis |

Context only, not yet fully verified: Mali et al. arXiv:2505.17893 (multicentre NSCLC radiomics + FM features); Nguyen et al. Front Artif Intell 2026 doi:10.3389/frai.2026.1865458 (LUNG1, C-index 0.641; Table 4 compiles prior LUNG1 C-indices); Paolo et al. Health Inf Sci Syst 2026 doi:10.1007/s13755-025-00404-z (authors' prior LUNG1 work); UCLIF, Radiology: Imaging Cancer 2026 doi:10.1148/rycan.250360 (cohorts unverified because of HTTP 403).

No survival or prognosis evaluation was found for COLIPRI, TANGERINE or Rad-JEPA 3D, including in their own papers.

**Novelty claims contradicted:** first FM survival evaluation on LUNG1/NSCLC-Radiogenomics; first external test on NSCLC-Radiogenomics; first DINOv3 or CT-CLIP on LUNG1; first whole-volume vs tumour-region FM comparison on LUNG1; first frozen vs LoRA vs fine-tuning CT FM for lung cancer survival; first tumour-dependence analysis.

**Defensible with "to our knowledge / we did not identify":** survival evaluation of COLIPRI, TANGERINE, Rad-JEPA 3D; a single protocol combining Cox/Uno C readouts, whole-volume FMs alongside lesion-encoder/CNN/radiomics/clinical baselines, label-budget curves, LoRA/partial tuning, test-time native-token tumour masking, and external calibration.

### Pretraining-overlap audit

| Model (local checkpoint) | Pretraining data | LUNG1 | NSCLC-Radiogenomics | Evidence | Confidence |
|---|---|---|---|---|---|
| COLIPRI (`external/colipri/model.safetensors`) | CT-RATE (~22.7k train volumes) + NLST (~72k, vision-only MAE objective); card: "trained with data from Turkey and the USA only" | Not listed | Not in pretraining. Downstream segmentation used MSD Task06 Lung (N=64), which MSD describes as the Stanford NSCLC radiogenomics cases; separate runs, not stated to affect the released encoder | arXiv:2510.15042v2 App. A.1, §5.5; HF card; MSD arXiv:1902.09063 | High / medium-high |
| TANGERINE (`external/tangerine/tangerine.pth`) | 98,593 volumes: SUMMIT (12.3%) + 27 public sets incl. CT-RATE, NLST-LSS, LIDC-IDRI, Lung-PET-CT-Dx, RIDER Lung CT and PET-CT, QIN Lung CT, CMB-LCA, LCTSC (60 series, NL/US), "NSCLC-Radiomics-Genomics – Lung 3" (MAASTRO, 54 series) | Not listed. Same-institution MAASTRO data present (Lung3, possibly LCTSC); patient-level overlap of LCTSC with LUNG1 **not determinable** | Not listed | Commun Med 2026 Methods + Supplementary Data 1 (MOESM3_ESM.xlsx) | Medium / high |
| Rad-JEPA 3D (`external/Rad-Jepa-3D-hf/encoder/encoder.pt`) | M3D-Cap, ~120k CT volumes from Radiopaedia (115k train) | Not listed; Radiopaedia provenance not auditable | Not listed | arXiv:2607.26196; M3D arXiv:2404.00578; HF card | Medium |
| CT-CLIP (`CT-CLIP_v2.pt`, third-party mirror Gayatrii/CTVIT) | CT-RATE, Istanbul Medipol (20,000 training patients) | No | No | Nat Biomed Eng 2026, doi:10.1038/s41551-025-01599-y | High (identity of mirrored checkpoint unverified) |
| DINOv3 ViT-B/16 LVD-1689M | Instagram web images (LVD-1689M) + ImageNet/Mapillary; ViT-B distilled from 7B | Implausible; not documentable | Implausible; not documentable | arXiv:2508.10104; HF card | High (by implausibility) |
| FMCIB (Pai) | DeepLesion (NIH) | Not in pretraining; prior external test set | Not in pretraining; prior external test set | pai2024 Use case 3, Data availability | High |
| Hosny 2018 | Supervised HarvardRT/Radboud | Test only (Maastro n=307) | Not used | hosny2018 | High |

Open items: (1) LCTSC MAASTRO cases vs LUNG1 patient IDs (check TCIA LCTSC metadata and DICOM UIDs/acquisition dates against LUNG1); (2) mapping of MSD Task06 cases to our external cohort subjects; (3) whether the CT-CLIP mirror matches the gated original (SHA-256 against the authors' file once access is granted); (4) explicit confirmation that the COLIPRI HF weights are the pretraining-only encoder; (5) confirm that Neukirch's "LUNG2" is NSCLC-Radiogenomics in the full text; (6) confirm Allgöwer's full-text numbers.
