# What Makes a Chest CT Foundation Model Prognostic?
## Benchmarking Prognostic Transferability and Survival-Specific Adaptation Across Pretraining Paradigms

**Research proposal — version 1.0**  
**Primary domain:** pretreatment chest CT in non-small cell lung cancer (NSCLC)  
**Primary endpoint:** Overall Survival (OS), modeled as a right-censored time-to-event outcome  
**Primary scientific focus:** properties of pretrained representations, not merely downstream predictive performance

---

## 1. Executive summary

Foundation models (FMs) for volumetric chest CT are increasingly pretrained using substantially different objectives: masked image reconstruction, vision-language alignment, joint-embedding prediction, and multi-objective learning. Existing studies have demonstrated that pretrained CT representations can support lung-cancer prognostic modeling. In particular, prior work has shown that:

- frozen CT foundation representations can contain prognostic information;
- pretrained CT encoders can be adapted to lung-cancer prognosis;
- Cox-type objectives can be combined with CT foundation representations;
- LoRA, frozen-backbone adaptation and full fine-tuning have already been evaluated for lung-cancer survival in individual foundation models;
- prognosis-oriented adaptation under limited supervision has been explored by ProtoSurv.

However, the current literature is predominantly **model-specific and performance-oriented**. It does not establish whether representations arising from fundamentally different pretraining paradigms exhibit systematically different:

1. **prognostic decodability** — how much survival information is already accessible from the frozen representation;
2. **prognostic adaptability** — how much additional survival information becomes accessible after a small survival-specific transformation;
3. **label/event efficiency** — how adaptation behaves as the number of observed survival events decreases;
4. **spatial organization of prognostic information** — whether the relevant signal is concentrated in tumor, peritumoral tissue or global thoracic context;
5. **representation transformation requirements** — how much the original feature geometry must change to support survival prediction;
6. **robustness and external transfer** — whether these properties persist across cohorts.

This project therefore treats **prognostic transferability as a property of the pretrained representation**, rather than treating a foundation model simply as an initialization for another downstream predictor.

The central question is:

> **How do the properties of general-purpose chest CT foundation representations determine their ability to support and adapt to overall-survival prediction under limited event supervision?**

The study will benchmark multiple modern chest CT foundation models representing distinct pretraining paradigms under a common survival-analysis protocol. A lightweight, common **Survival Query Adapter (SQA)** will be used as an experimental probe to quantify how easily each pretrained representation can be reorganized into a survival-specific latent space. The adapter is not required to outperform every alternative adaptation method for the study to succeed; its primary role is to provide a controlled and comparable transformation mechanism across heterogeneous representations.

---

# 2. Scientific motivation

## 2.1 General-purpose representations are not necessarily prognostic representations

Modern CT foundation models are typically optimized for objectives that encourage broad anatomical, semantic or diagnostic representations. Examples include:

- reconstruction of masked CT patches;
- alignment between CT volumes and radiology reports;
- prediction of latent representations of missing regions;
- combinations of visual, textual and dense prediction objectives.

None of these objectives directly optimizes the geometry of the latent space for censored time-to-event prediction.

A representation can therefore be highly useful for diagnosis while displaying very different behavior for prognosis. For survival prediction, useful information may include not only tumor appearance but also weak distributed signals reflecting disease burden, nodal disease, surrounding parenchyma, body composition, comorbidities, treatment-relevant anatomy and other contextual phenotypes.

The proposed study asks whether such prognostic information is:

- already explicitly encoded and linearly accessible;
- present but distributed or entangled in the latent space;
- recoverable only after nonlinear adaptation;
- absent or substantially degraded by a specific representation-learning strategy.

This distinction motivates the separation between **decodability** and **adaptability**.

---

# 3. Literature gap

## 3.1 What has already been demonstrated

The project does **not** claim that CT foundation models have never been applied to survival prediction. Several important precedents exist.

### Pai et al. — Foundation model for cancer imaging biomarkers

Pai et al. developed a self-supervised CT foundation model and evaluated frozen and fine-tuned representations for several cancer imaging tasks. For NSCLC prognostication they trained a 2-year overall-survival classifier using HarvardRT and externally evaluated on LUNG1 and NSCLC-Radiogenomics. Frozen foundation features followed by a shallow classifier were highly competitive, establishing that pretrained CT representations can already encode prognostic information.

**Reference:**  
Pai S, Bontempi D, Hadzic I, et al. *Foundation model for cancer imaging biomarkers.* Nature Machine Intelligence. 2024;6:354–367.  
https://doi.org/10.1038/s42256-024-00807-9

### LCTfound

LCTfound is a lung CT foundation model pretrained using a diffusion-based strategy on a large lung CT collection. The study includes NSCLC prognosis and compares frozen-feature and fine-tuning strategies, demonstrating that foundation representations can be transferred to lung-cancer prognostic tasks.

**Reference:**  
Gao Z, Zhang G, Liang H, et al. *A lung CT vision foundation model facilitating disease diagnosis and medical imaging.* Nature Communications. 2025.  
https://doi.org/10.1038/s41467-025-66620-z

### CT-CLIP survival study

A 2026 MICCAI/TIA study directly evaluated CT-CLIP representations for lung-cancer survival prediction in 242 patients. It compared a frozen encoder, full fine-tuning and low-rank adaptation (LoRA) using survival modeling. This establishes that **frozen vs PEFT vs full adaptation for lung-cancer survival is not itself a novel research question**.

**Reference:**  
Allgöwer S, Johansson M, Hallqvist A, et al. *CT-CLIP Representations for Multimodal Lung Cancer Survival Prediction.* MICCAI 2026 Workshops and Challenges / Thoracic Image Analysis workshop.  
https://arxiv.org/abs/2607.08503  
https://papers.miccai.org/miccai-2026-sat/TIA_007.html

### ProtoSurv

ProtoSurv proposes prognosis-oriented adaptation of a CT vision-language model under limited labeled supervision. It uses prognosis prototypes, pseudo-label assignment, confidence filtering and feature calibration, exploiting large-scale unlabeled CT data. It evaluates both PFS and OS in lung cancer and represents the closest existing work to the general idea of adapting pretrained CT representations for prognosis.

**Reference:**  
Liu C, Li S, Sang H, et al. *ProtoSurv: prototype-guided adaptation of computed tomography foundation model for lung-cancer prognosis prediction.* Visual Computing for Industry, Biomedicine, and Art. 2026;9:21.  
https://doi.org/10.1186/s42492-026-00233-1

### PRISM2 — analogous evidence outside CT

In computational pathology, PRISM2 provides a particularly relevant conceptual precedent. It learns new cross-attention pooling specifically for survival and creates survival-specialized embeddings optimized using Cox partial likelihood. Therefore, the generic concept of converting general foundation embeddings into survival-specific embeddings using attention is not novel in isolation.

**Reference:**  
*End-to-end multimodal pathology foundation model with clinical dialogue.* Nature Medicine. 2026.  
https://doi.org/10.1038/s41591-026-04521-4

---

## 3.2 The unresolved gap

The previous studies establish that foundation representations can support prognosis, but they leave a different question unanswered:

> **Are foundation representations learned through different pretraining paradigms equally prognostically transferable, and what representation properties explain their ability to adapt to survival prediction?**

To our knowledge, there is no systematic study in lung-cancer CT that simultaneously:

- evaluates multiple modern volumetric CT foundation models arising from substantially different pretraining paradigms;
- applies the **same censoring-aware OS endpoint** and the same downstream evaluation protocol;
- separates **frozen prognostic decodability** from **adaptation gain**;
- measures adaptation as a function of the amount of observed survival supervision;
- studies where prognostic information is localized within the volumetric representation;
- compares latent geometry before and after survival adaptation;
- evaluates whether adaptation amplifies genuine prognostic signal or merely stage, tumor size or technical confounding;
- performs external validation of these representation-level properties.

Accordingly, the novelty is **not**:

> “using a CT foundation model for survival prediction.”

Nor is it:

> “using Cox loss with a pretrained CT encoder.”

Nor is it:

> “comparing frozen, LoRA and fine-tuned models.”

The novelty is instead the **systematic characterization of prognostic transferability and adaptability as measurable properties of heterogeneous pretrained CT representations**.

---

# 4. Main research question

> **What properties of chest CT foundation-model representations determine their prognostic transferability for overall survival in NSCLC?**

Secondary questions:

1. How much OS information is already accessible from each frozen representation?
2. Are representations that are highly decodable also highly adaptable?
3. Does the pretraining paradigm correlate with different patterns of prognostic decodability and adaptability?
4. Does survival-specific adaptation become more beneficial when event supervision is limited?
5. Is prognostic signal localized primarily within tumor tokens, peritumoral tissue or global thoracic context?
6. Does adaptation learn survival-specific information beyond stage and tumor burden?
7. How much does the latent geometry change during adaptation?
8. Do these representation properties generalize to an external cohort?

---

# 5. Foundation models

The core benchmark will contain four modern volumetric chest CT foundation models.

## 5.1 TANGERINE — masked autoencoding

**Paradigm:** self-supervised 3D masked autoencoding / voxel-space reconstruction.

TANGERINE was pretrained on 98,588 thoracic low-dose CT volumes using a 3D masked autoencoder and was designed for computationally efficient downstream adaptation.

**Role in the study:** representative of reconstruction-based masked image modeling.

**Reference:**  
*A computationally frugal, open-source chest CT foundation model for thoracic disease detection in lung cancer screening programmes.* Communications Medicine. 2025.  
https://doi.org/10.1038/s43856-025-01328-1

---

## 5.2 CT-CLIP — vision-language contrastive pretraining

**Paradigm:** CT-report contrastive / vision-language representation learning.

**Role in the study:** representative of language-aligned semantic pretraining and, importantly, a positive methodological control because it has already been evaluated for lung-cancer survival.

**Reference:**  
https://arxiv.org/abs/2607.08503

---

## 5.3 Rad-JEPA-3D — joint-embedding predictive learning

**Paradigm:** latent-space prediction from masked volumetric views using a joint-embedding predictive architecture.

Rad-JEPA-3D was pretrained on approximately 120,000 CT scans and explicitly aims to preserve volumetric and spatial structure.

**Role in the study:** representative of latent predictive learning, conceptually distinct from both reconstruction and image-text alignment.

**Reference:**  
Trinh Q-H, Nguyen M-V, Bagci U. *Rad-JEPA 3D: Radiology Joint-Embedding Predictive Model for 3D Computed Tomography.* 2026.  
https://arxiv.org/abs/2607.26196  
https://huggingface.co/huyquoctrinh/Rad-Jepa-3D

---

## 5.4 COLIPRI — multimodal / multi-objective CT representation

**Paradigm:** 3D vision-language representation learning using the Primus-M visual encoder, with CT and radiology-report supervision and dense volumetric representations.

**Role in the study:** representative of large-scale semantic/multi-objective CT representation learning with explicit dense volumetric features.

**Reference / model:**  
Microsoft COLIPRI model card.  
https://huggingface.co/microsoft/colipri  
https://huggingface.co/microsoft/colipri/blob/main/model_card.md

---

## 5.5 Optional reference model: FMCIB

FMCIB will be considered as an optional fifth/reference encoder because it has already been validated on LUNG1 and NSCLC-Radiogenomics. It will not necessarily be included in all token-level analyses because its architecture and lesion-centric representation differ substantially from the four main volumetric encoders.

**Reference:** Pai et al., Nature Machine Intelligence 2024.

---

# 6. Important interpretation constraint

The benchmark compares **foundation models arising from different pretraining paradigms**, but it does not isolate the causal effect of the pretraining objective.

The models also differ in:

- architecture;
- model size;
- tokenization;
- training datasets;
- amount of pretraining data;
- input resolution;
- preprocessing;
- optimization and regularization.

Therefore, conclusions will be formulated as:

> “Foundation models arising from different pretraining paradigms exhibit different prognostic transferability profiles.”

They will **not** be formulated as:

> “JEPA causally produces more prognostic representations than masked autoencoding.”

A causal comparison of objectives would require controlled pretraining using the same architecture, dataset, compute budget and augmentation pipeline, which is outside the scope of the primary study.

A smaller controlled auxiliary experiment using matched architecture/objective variants may be added if suitable public checkpoints become available.

---

# 7. Data

## 7.1 Primary cohort — NSCLC-Radiomics / LUNG1

- 422 NSCLC subjects;
- pretreatment chest CT;
- manual gross tumor volume (GTV) delineations;
- clinical information;
- overall-survival time;
- death/censoring indicator.

Survival time is measured from treatment start, and the event indicator follows standard right-censoring conventions.

**Primary source:**  
The Cancer Imaging Archive / Imaging Data Commons, NSCLC-Radiomics.  
DOI: 10.7937/K9/TCIA.2015.PF0M9REI  
https://www.cancerimagingarchive.net/collection/nsclc-radiomics/

The final analytic population will be defined prospectively after data-quality control. Cases with invalid outcome data, unusable CT series or irrecoverable preprocessing failures will be excluded according to predefined criteria.

---

## 7.2 External cohort — NSCLC-Radiogenomics

- 211 total NSCLC subjects in the complete collection;
- pretreatment CT;
- clinical follow-up and survival outcomes;
- tumor annotations/segmentations for a subset.

**Reference:**  
Bakr S, Gevaert O, Echegaray S, et al. *A radiogenomic dataset of non-small cell lung cancer.* Scientific Data. 2018;5:180202.  
https://doi.org/10.1038/sdata.2018.202  
TCIA: https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/

Because not all 211 subjects necessarily have all required combinations of CT, segmentation and valid OS information, the effective external sample size will be determined after harmonized eligibility filtering and reported explicitly.

---

# 8. Conceptual framework: properties of prognostic representations

The study defines six primary representation properties.

---

## Property P1 — Prognostic decodability

**Definition:** how much prognostic information is already accessible from the frozen representation using a low-capacity survival model.

For foundation model \(m\), frozen embedding \(Z_m\), and a linear Cox head:

\[
D_m =
C_{\mathrm{Uno}}
\left(
\mathrm{LinearCox}(Z_m)
\right)
\]

where \(C_{\mathrm{Uno}}\) is the primary concordance measure.

### Hypothesis H1

> Different CT foundation models will exhibit significantly different prognostic decodability despite all providing useful general-purpose representations.

### Experiment E1 — frozen representation audit

For every FM:

1. freeze the complete encoder;
2. extract global pooled embeddings;
3. fit the same regularized linear Cox model;
4. evaluate identical folds;
5. compare C-index, IPCW time-dependent AUC, IBS and calibration.

Additional probes:

- stage;
- T/N category where available;
- tumor volume;
- histology;
- age/sex;
- acquisition-related variables where recoverable.

The objective is to determine what information is already linearly accessible from each representation.

---

## Property P2 — Prognostic adaptability

**Definition:** the gain obtained when a frozen representation is allowed to undergo a small survival-specific nonlinear transformation.

\[
A_m =
C_{\mathrm{Uno}}
\left(
\mathrm{SQA}(Z_m)
\right)
-
C_{\mathrm{Uno}}
\left(
\mathrm{LinearCox}(Z_m)
\right)
\]

### Hypothesis H2

> Prognostic decodability and prognostic adaptability are distinct representation properties: a representation with strong frozen performance will not necessarily obtain the largest gain from survival-specific adaptation.

This permits at least four qualitative regimes:

| Decodability | Adaptability | Interpretation |
|---|---|---|
| high | low | survival information is already accessible |
| high | high | representation is both informative and highly reorganizable |
| low | high | useful information is latent/entangled but recoverable |
| low | low | representation contains limited transferable prognostic signal |

### Experiment E2 — common adaptation ladder

For every FM, use the same downstream hierarchy:

1. frozen global embedding + Linear Cox;
2. frozen embedding + MLP survival head;
3. frozen token representation + Survival Query Adapter;
4. LoRA where architecturally meaningful;
5. partial fine-tuning;
6. optional full fine-tuning.

Primary comparison:

\[
\text{Linear} \rightarrow \text{SQA}
\]

Secondary comparisons quantify parameter efficiency relative to LoRA and fine-tuning.

---

## Property P3 — Event efficiency

**Definition:** how effectively prognostic information can be learned as the amount of observed survival supervision decreases.

Because information in survival analysis depends strongly on the number of events, not only on the number of patients, the study will report both.

### Hypothesis H3

> Foundation models will differ more strongly in prognostic adaptability under limited event supervision than when the full training cohort is available.

A second sub-hypothesis is:

> The lightweight Survival Query Adapter will yield a larger relative benefit over higher-capacity fine-tuning strategies in the lowest-supervision regimes.

### Experiment E3 — event/label-efficiency curves

Within every training fold, generate nested training subsets corresponding approximately to:

- 10%;
- 25%;
- 50%;
- 100%

of the available training subjects, while preserving the event/censoring distribution as far as possible.

For each subset report:

- number of patients;
- number of death events;
- censoring fraction;
- C-index;
- IBS;
- AUC at predefined clinically meaningful horizons.

Main visualization:

\[
\text{performance}
\quad \text{vs} \quad
N_{\mathrm{events}}
\]

For each model, estimate an event-efficiency summary such as normalized area under the learning curve.

---

## Property P4 — Spatial prognostic organization

**Definition:** where prognostic information is represented spatially within volumetric FM tokens.

Define token sets:

\[
Z =
\{
Z_{\mathrm{tumor}},
Z_{\mathrm{peri}},
Z_{\mathrm{global}}
\}
\]

using the GTV segmentation projected onto the token grid.

Candidate peritumoral rings:

- 0–5 mm;
- 5–10 mm;
- 10–20 mm;

with sensitivity analyses because token resolution differs across FMs.

### Hypothesis H4

> Different foundation representations distribute prognostic information differently across tumor, peritumoral tissue and global thoracic context.

A secondary hypothesis is:

> Survival adaptation will increase the contribution of contextual/peritumoral information in models whose pretraining objective preserves volumetric or semantic context.

This is exploratory with respect to the direction of the effect; the primary confirmatory claim is that spatial contribution profiles differ between models.

### Experiment E4 — controlled regional ablation

For each FM evaluate:

- tumor tokens only;
- peritumoral tokens only;
- tumor + peritumor;
- global tokens excluding tumor;
- all tokens.

Measure performance using both:

1. frozen region-pooled linear probes;
2. Survival Query Adapter.

Compare regional attribution before and after adaptation.

---

## Property P5 — Representation transformation

**Definition:** how much the latent geometry must change to support survival prediction.

### Hypothesis H5

> Models with lower frozen decodability but high adaptation gain will exhibit larger task-specific changes in representation geometry than models whose prognostic signal is already linearly accessible.

### Experiment E5 — representation geometry before vs after adaptation

Possible analyses:

- centered kernel alignment (CKA);
- representational similarity analysis;
- cosine neighborhood preservation;
- effective rank / spectral concentration;
- within-risk and between-risk neighborhood structure;
- risk-gradient alignment;
- token attention redistribution.

For a representation \(Z^{pre}\) and adapted representation \(Z^{surv}\):

\[
\Delta R_m =
1-
\mathrm{CKA}(Z^{pre}_m,Z^{surv}_m)
\]

or an equivalent normalized representation-shift measure.

Correlate:

\[
\Delta R_m
\]

with:

\[
A_m
\]

to test whether larger adaptation gains require larger representation shifts.

---

## Property P6 — Prognostic specificity vs clinical/confounding information

A survival representation may improve because it learns genuine prognostic phenotypes, but it may also simply become a proxy for stage, tumor size or acquisition/site artifacts.

### Hypothesis H6

> Survival adaptation will increase decodability of OS more than it increases decodability of nuisance variables, and at least part of the prognostic gain will remain after adjustment for established clinical covariates.

### Experiment E6 — conditional prognostic value

For each representation compare:

\[
\text{Clinical Cox}
\]

versus

\[
\text{FM risk only}
\]

versus

\[
\text{Clinical Cox + FM risk}.
\]

Candidate clinical variables, subject to harmonized availability:

- age;
- sex;
- stage;
- T category;
- N category;
- histology;
- tumor volume.

Evaluate:

- likelihood-ratio improvement;
- C-index improvement;
- IBS improvement;
- calibration;
- multivariable hazard ratio for FM risk.

Also probe technical variables when available:

- reconstruction kernel;
- slice thickness;
- scanner manufacturer/model;
- voxel spacing.

The goal is not to prove complete biological specificity but to quantify the extent to which survival adaptation is separable from known clinical and technical correlates.

---

# 9. Proposed Survival Query Adapter (SQA)

## 9.1 Purpose

The SQA is primarily a **standardized adaptation probe** rather than the sole source of novelty.

It should:

- accept heterogeneous token representations;
- keep the FM backbone frozen;
- use few trainable parameters;
- operate directly on spatial token sequences;
- use censoring-aware survival supervision;
- provide interpretable token/query attention;
- enable identical adaptation logic across models.

---

## 9.2 Model-specific input normalization

For foundation model \(m\):

\[
Z_m \in \mathbb{R}^{N_m \times d_m}
\]

Project all representations to a common adapter dimension:

\[
\tilde Z_m =
\mathrm{LayerNorm}(Z_m W_m)
\]

with:

\[
\tilde Z_m \in \mathbb{R}^{N_m \times d}
\]

A candidate value is:

\[
d = 256
\]

but \(d\) will be fixed before primary comparison.

---

## 9.3 Learnable prognostic queries

Define:

\[
Q \in \mathbb{R}^{K\times d}
\]

where \(K\) is a small number of learnable survival queries, e.g. \(K \in \{4,8\}\).

Cross-attention:

\[
H =
\mathrm{CrossAttention}
(
Q,
\tilde Z_m,
\tilde Z_m
)
\]

followed by one or two low-capacity transformer/MLP blocks.

Aggregate:

\[
z_{\mathrm{surv}}
=
\mathrm{Pool}(H)
\]

and predict log-risk:

\[
r =
w^\top z_{\mathrm{surv}} + b.
\]

---

## 9.4 Primary loss

Primary objective:

\[
\mathcal L_{\mathrm{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
r_i
-
\log
\sum_{j:t_j\geq t_i}
\exp(r_j)
\right].
\]

Ties will be handled using a predefined approximation such as Efron.

The primary study will avoid unnecessary auxiliary losses unless they are supported by validation experiments.

---

## 9.5 Optional secondary objectives

Only after the main Cox-SQA has been established:

### Pairwise survival ranking

\[
\mathcal L =
\mathcal L_{\mathrm{Cox}}
+
\lambda_{\mathrm{rank}}
\mathcal L_{\mathrm{rank}}
\]

provided comparisons respect censoring and comparable risk sets.

### Discrete-time survival

As a robustness experiment, model interval-specific hazards and compare representation conclusions against the Cox formulation.

The research conclusions should not depend on one specific survival loss.

---

# 10. Region-aware Survival Query Adapter

A secondary extension will use explicit regional query banks:

\[
Q_T,\quad Q_P,\quad Q_G
\]

for:

- tumor;
- peritumoral tissue;
- global context.

Then:

\[
H_T =
\mathrm{CrossAttn}(Q_T,Z_T,Z_T)
\]

\[
H_P =
\mathrm{CrossAttn}(Q_P,Z_P,Z_P)
\]

\[
H_G =
\mathrm{CrossAttn}(Q_G,Z_G,Z_G)
\]

and:

\[
z_{\mathrm{surv}}
=
\mathrm{Fuse}(H_T,H_P,H_G).
\]

This model will **not** be the primary adapter because it receives privileged segmentation information. It is an explanatory extension aimed at understanding spatial organization of prognostic information.

---

# 11. Experimental pipeline

## Stage 0 — Data curation

1. Download LUNG1 and external validation data.
2. Identify one pretreatment chest CT per subject using predefined selection rules.
3. Validate survival time and event indicator.
4. Validate image orientation, dimensions and DICOM consistency.
5. Convert segmentations to a common coordinate representation.
6. Compute tumor volume from GTV.
7. Define a locked eligibility report before model comparison.

No model-dependent case exclusion will be permitted unless technically unavoidable and explicitly reported.

---

## Stage 1 — Model-specific preprocessing

Each FM will first use its **official recommended preprocessing pipeline** to avoid artificially degrading a model through incompatible input construction.

Record for every model:

- HU clipping/window;
- voxel spacing;
- resize/crop policy;
- number of slices;
- intensity normalization;
- spatial field of view;
- token-grid resolution.

A secondary harmonization analysis may use the most comparable common field of view possible, but the primary benchmark should respect each pretrained model's intended input domain.

---

## Stage 2 — Frozen feature extraction

For each patient and model save:

1. global embedding;
2. final-layer token embeddings;
3. selected intermediate-layer tokens where accessible.

All downstream frozen experiments should operate on cached embeddings to reduce computational cost and guarantee deterministic representation inputs.

---

## Stage 3 — Representation audit

### Global probes

- linear Cox OS;
- regularized Cox OS;
- optional shallow MLP survival head.

### Clinical probes

Predict:

- stage;
- T;
- N;
- histology;
- tumor volume;
- age/sex.

### Technical probes

Predict available acquisition characteristics.

### Layer-wise analysis

If hidden states are available:

\[
Z^{(1)}, Z^{(2)}, ..., Z^{(L)}
\]

fit identical low-capacity probes at several depths.

This tests whether prognostic information emerges early, gradually or primarily in the final representation.

---

## Stage 4 — Adaptation benchmark

For each FM:

### A. Linear Cox

Frozen representation.

### B. MLP Cox

Frozen representation + nonlinear head.

### C. Survival Query Adapter

Frozen backbone + token-level SQA.

### D. LoRA

Apply a matched low-rank adaptation strategy where architecture permits.

### E. Partial fine-tuning

Unfreeze the last block(s) or predefined percentage of the encoder.

### F. Full fine-tuning

Optional because of computational cost and overfitting risk.

For all methods report:

- trainable parameter count;
- percentage of total parameters;
- peak GPU memory;
- training time;
- convergence stability.

---

## Stage 5 — Event-efficiency experiment

Repeat A–E with:

- 10%;
- 25%;
- 50%;
- 100%

of the training data.

The *same sampled subject subsets* must be used across models within each fold/repetition.

Primary x-axis:

\[
N_{\mathrm{observed\ events}}.
\]

---

## Stage 6 — Regional representation audit

Project GTV and peritumoral regions onto each model's token grid.

Perform:

- regional frozen Cox probes;
- token masking/ablation;
- SQA attention analysis;
- region-aware SQA.

This answers whether prognostic transferability is associated with spatially distinct representation patterns.

---

## Stage 7 — Representation-shift analysis

For each model:

1. extract \(Z_{\mathrm{pre}}\);
2. extract \(Z_{\mathrm{surv}}\) after SQA;
3. quantify similarity/shift;
4. test association between representation shift and adaptation gain.

---

## Stage 8 — Clinical incremental-value analysis

Fit within each training split:

### Model C

\[
h(t|x_c)
=
h_0(t)\exp(\beta_c^\top x_c)
\]

### Model F

\[
h(t|r)
=
h_0(t)\exp(\beta_r r)
\]

### Model C+F

\[
h(t|x_c,r)
=
h_0(t)\exp(\beta_c^\top x_c+\beta_r r).
\]

Compare on held-out data.

---

## Stage 9 — External validation

Lock:

- preprocessing;
- hyperparameters;
- adapter architecture;
- trained model or training protocol;
- risk-score scaling.

Evaluate on NSCLC-Radiogenomics.

Because treatment context and cohort composition differ from LUNG1, external validation will be interpreted as **transportability testing**, not merely another random test split.

---

# 12. Experimental matrix

## 12.1 Core models

| Model | Primary pretraining paradigm | Global representation | Token representation | Core benchmark |
|---|---|---:|---:|---:|
| TANGERINE | 3D masked autoencoding | yes | yes | yes |
| CT-CLIP | vision-language contrastive | yes | architecture-dependent extraction | yes |
| COLIPRI | 3D vision-language / multi-objective | yes | yes | yes |
| Rad-JEPA-3D | joint-embedding predictive learning | yes | yes | yes |
| FMCIB | CT SSL / lesion-centric reference | yes | limited/non-equivalent | optional |

---

## 12.2 Adaptation methods

| Method | Backbone | Token-level? | Expected parameter budget | Scientific role |
|---|---|---:|---:|---|
| Linear Cox | frozen | no | very low | decodability |
| MLP Cox | frozen | no | low | nonlinear pooled baseline |
| SQA | frozen | yes | low | adaptability probe |
| LoRA | mostly frozen | yes/model-dependent | low-medium | PEFT baseline |
| Partial FT | partly trainable | yes | medium | adaptation-capacity baseline |
| Full FT | trainable | yes | high | upper-capacity reference |

---

# 13. Primary and secondary endpoints

## Primary metric

### Uno's concordance index

Preferred over naïve Harrell C when censoring is substantial.

Report:

- point estimate;
- bootstrap 95% confidence interval;
- paired differences between methods/models.

---

## Secondary metrics

### Time-dependent AUC

At predefined horizons such as:

- 12 months;
- 24 months;
- 36 months;

subject to adequate at-risk counts.

### Integrated Brier Score

Measures discrimination and calibration jointly over a predefined evaluation interval.

### Calibration

- calibration slope;
- calibration-in-the-large where applicable;
- predicted vs observed survival probability;
- integrated calibration error if appropriate.

### Kaplan–Meier risk stratification

Only secondary/explanatory.

Risk thresholds must be determined exclusively from training data.

---

# 14. Validation design

Because LUNG1 is small for deep learning, random single-split evaluation is inadequate.

Recommended primary protocol:

## Repeated nested cross-validation

Example:

- outer: 5 folds;
- repetitions: 5;
- inner: 3–5 folds for hyperparameter tuning.

All models and adaptation methods use **exactly the same outer-fold assignments**.

Stratification should approximately preserve:

- event indicator;
- stage distribution if possible.

For survival data, care must be taken that each validation fold contains sufficient events.

External NSCLC-Radiogenomics remains fully untouched until the primary model-development protocol is locked.

---

# 15. Statistical analysis

## 15.1 Paired comparisons

Because every model is evaluated on the same patients/folds, comparisons will be paired.

Primary comparisons:

1. differences in frozen Linear Cox performance between FMs;
2. SQA vs Linear Cox within each FM;
3. adaptation-gain differences between FMs;
4. low-event vs full-data adaptation gain.

Use:

- paired patient-level bootstrap where appropriate;
- fold/repetition-aware confidence intervals;
- corrected resampled tests only if their assumptions are defensible.

Avoid significance testing on individual CV folds as if folds were independent observations.

---

## 15.2 Multiple comparisons

Predefine a small set of confirmatory contrasts.

For example:

- H1: between-model frozen representation comparison;
- H2: Linear Cox vs SQA within each FM;
- H3: interaction between adaptation strategy and supervision regime.

Use Holm correction for the limited family of confirmatory comparisons.

All remaining pairwise exploration should be labeled exploratory.

---

## 15.3 Survival-model assumption checks

For Cox-based clinical models:

- inspect proportional-hazards assumptions;
- consider time-varying effects if clearly violated.

The neural risk score does not remove the proportional-hazards interpretation of a Cox loss; this limitation must be acknowledged.

Discrete-time survival will provide a sensitivity analysis.

---

# 16. Main hypotheses and decision rules

## H1 — Foundation representations differ in prognostic decodability

**Null:** frozen representations yield equivalent out-of-sample prognostic discrimination after accounting for uncertainty.

**Support for H1:** reproducible differences in held-out C-index/IBS across FMs with paired uncertainty estimates.

---

## H2 — Decodability and adaptability are distinct

**Null:** models with stronger frozen performance necessarily retain the same ordering after lightweight adaptation, and adaptation gains do not meaningfully differ.

**Support for H2:** different models exhibit statistically and practically different:

\[
(D_m,A_m)
\]

profiles.

Primary visualization: **prognostic representation map**

- x-axis: frozen prognostic decodability;
- y-axis: adaptation gain.

---

## H3 — Adaptability differs under limited event supervision

**Null:** relative differences between representations and adaptation methods are stable across supervision levels.

**Support for H3:** model-by-supervision or adaptation-by-supervision interactions, with clearly different learning curves.

---

## H4 — Prognostic information has model-specific spatial organization

**Null:** tumor/peritumor/global contribution profiles are equivalent across foundation representations.

**Support for H4:** consistent cross-model differences in region-restricted probes/ablations.

No directional assumption is required a priori.

---

## H5 — Adaptation gain is associated with representational transformation

**Null:** adaptation gain is unrelated to latent-space change.

**Support for H5:** reproducible association between representation-shift measures and prognostic gain.

---

## H6 — Adapted representations contain prognostic information beyond simple clinical proxies

**Null:** FM risk provides no incremental held-out prognostic value after conditioning on available established clinical variables.

**Support for H6:** consistent improvement of Clinical + FM over Clinical alone in held-out evaluation and external validation.

---

# 17. Primary figures

## Figure 1 — Study concept

\[
CT
\rightarrow
FM
\rightarrow
Z_{general}
\rightarrow
Survival\ Adaptation
\rightarrow
Z_{surv}
\rightarrow
Risk
\]

Four pretraining paradigms enter the same downstream evaluation protocol.

---

## Figure 2 — Prognostic representation map

Scatter plot:

- x = frozen decodability;
- y = adaptation gain;
- point = foundation model;
- size or annotation = trainable parameter count / pretraining paradigm.

This is proposed as the central conceptual figure.

---

## Figure 3 — Event-efficiency curves

For each FM:

\[
C\text{-index}
\;\text{vs}\;
N_{\mathrm{events}}
\]

for Linear Cox, SQA and LoRA.

---

## Figure 4 — Spatial prognostic decomposition

Heatmap:

| Model | Tumor | Peritumor | Global | Tumor+Peri | All |
|---|---:|---:|---:|---:|---:|

showing region-restricted survival performance.

---

## Figure 5 — Representation transformation

For each FM:

- pretrained-adapted CKA;
- adaptation gain;
- relationship between representation shift and survival performance.

---

## Figure 6 — External transfer

Internal vs external:

- C-index;
- IBS;
- calibration;
- clinical + FM incremental value.

---

# 18. Baselines

The benchmark should include:

## Clinical baselines

- clinical Cox;
- stage-only Cox;
- clinical Cox + tumor volume.

## Classical imaging baseline

- PyRadiomics features + penalized Cox;
- optional random survival forest / survival boosting.

## FM baselines

- frozen FM + Linear Cox;
- frozen FM + MLP Cox;
- LoRA;
- partial fine-tuning;
- optional full fine-tuning.

## Literature-reference methods

ProtoSurv should be discussed and, where technically reproducible and fair, compared experimentally. However, direct reproduction may be difficult because ProtoSurv uses additional large-scale unlabeled data and its own pretraining/adaptation pipeline. If identical data resources are not available, it should not be presented as a strictly controlled head-to-head comparison.

---

# 19. Fairness of comparison across foundation models

This is a critical part of the study.

## 19.1 Same patients

All models must be evaluated on the same eligible patient set whenever possible.

If a model fails on a patient due to model-specific technical constraints, the primary paired analysis should either:

- use a common intersection cohort; or
- perform a predefined sensitivity analysis.

---

## 19.2 Same folds

All models use identical train/validation/test assignments.

---

## 19.3 Same survival objective

For a given experimental comparison, downstream loss and risk-set implementation must be identical.

---

## 19.4 Same adapter capacity where possible

The common SQA should use:

- same projection dimension;
- same number of queries;
- same number of attention blocks;
- approximately matched trainable parameter budget.

Only the input projection \(d_m \rightarrow d\) is model-specific.

---

## 19.5 No test-set tuning

No decisions about:

- preprocessing;
- hyperparameters;
- region definitions;
- risk threshold;
- adapter depth;

may be based on outer test performance or external validation.

---

# 20. Hyperparameter strategy

Given the small sample size, avoid large model-specific searches.

Predefine a compact shared space, e.g.:

- adapter dimension: 128 or 256;
- query number: 4 or 8;
- dropout: 0.1 or 0.2;
- learning rate: \(10^{-4}\), \(3\times10^{-4}\), \(10^{-3}\);
- weight decay: 0, \(10^{-4}\), \(10^{-3}\).

Use the same search space across FMs.

The project should measure representation properties, not reward the model receiving the largest bespoke hyperparameter search.

---

# 21. Potential failure modes and what they would mean

## Failure mode 1 — no FM differences

If all FMs have statistically indistinguishable frozen and adapted performance, this is still scientifically informative:

> modern large-scale chest CT pretraining paradigms may converge toward similarly useful prognostic representations despite different objectives.

---

## Failure mode 2 — SQA does not outperform MLP/LoRA

This does not invalidate the representation study.

If SQA fails to improve performance, the result indicates that:

- survival information is already accessible through pooled embeddings; or
- token-level reorganization provides limited additional value at LUNG1 scale.

The paper should not depend on SQA winning.

---

## Failure mode 3 — adaptation gains disappear externally

This would suggest that the adapter learns cohort-specific prognostic structure.

Representation robustness then becomes a central result.

---

## Failure mode 4 — gains vanish after stage/tumor-volume adjustment

This would indicate that FM prognosis is largely mediated by established disease-burden variables.

Again, this is scientifically meaningful and should be reported rather than hidden.

---

## Failure mode 5 — regional results depend strongly on token resolution

Perform sensitivity analyses using:

- physical-distance peritumoral regions;
- spatial interpolation onto a common coarse grid;
- multiple ring widths.

---

# 22. Expected contributions

The proposed work is expected to provide the following contributions.

## Contribution 1 — First controlled prognostic-transferability benchmark across modern chest CT FM paradigms

A unified survival-analysis comparison of representations arising from:

- masked reconstruction;
- vision-language contrastive learning;
- joint-embedding predictive learning;
- multi-objective / multimodal CT learning.

---

## Contribution 2 — Formal separation between prognostic decodability and adaptability

Instead of ranking models solely by final C-index, the study introduces a representation-oriented view:

\[
\text{prognostic utility}
=
\text{what is already accessible}
+
\text{what can be efficiently reorganized}.
\]

---

## Contribution 3 — Event-efficiency characterization

Quantify representation behavior as a function of actual survival supervision, emphasizing the number of observed events.

---

## Contribution 4 — Survival-specific adaptation probe

A lightweight common query-based adapter enabling token-level adaptation under frozen backbones.

Its value is primarily methodological and comparative; superiority to all PEFT methods is not a prerequisite for the main scientific conclusions.

---

## Contribution 5 — Spatial representation analysis

Determine whether prognostic information is encoded differently across:

- tumor;
- peritumoral tissue;
- global thoracic context.

---

## Contribution 6 — Representation-shift analysis

Measure how much the general-purpose latent geometry changes when optimized for survival and relate this change to adaptation gain.

---

## Contribution 7 — Incremental and external validity

Test whether learned prognostic representations provide information beyond clinical covariates and whether their properties transfer to an independent cohort.

---

# 23. Proposed paper positioning

## Recommended title

**What Makes a Chest CT Foundation Model Prognostic? Benchmarking Prognostic Transferability Across Pretraining Paradigms**

Alternative:

**From General CT Representations to Survival-Specific Subspaces: A Benchmark of Prognostic Transferability in Chest CT Foundation Models**

---

## Recommended central claim

> General-purpose chest CT foundation models should not be compared only by downstream survival performance. Their representations can differ independently in the amount of prognostic information they already expose, the amount recoverable through lightweight adaptation, their event efficiency, their spatial organization and the magnitude of representation transformation required for prognosis.

---

## Claim to avoid

> We are the first to adapt a CT foundation model to lung-cancer survival.

This is contradicted by prior work.

---

## Claim to avoid

> We are the first to use censoring-aware learning with foundation CT features.

This is also contradicted by prior work.

---

## Claim to avoid

> The pretraining objective causes the observed prognostic differences.

The benchmark is observational with respect to pretraining design because architecture, dataset and scale differ across FMs.

---

# 24. Minimal viable study vs full study

## Minimal viable paper

If compute or implementation time is constrained:

**Models**
- TANGERINE;
- CT-CLIP;
- COLIPRI;
- Rad-JEPA-3D.

**Experiments**
1. frozen Linear Cox;
2. frozen MLP Cox;
3. SQA;
4. 25/50/100% supervision;
5. tumor/peritumor/global ablation;
6. external validation.

This is sufficient to answer the principal scientific question.

---

## Full study

Add:

- LoRA;
- partial/full fine-tuning;
- 10% supervision;
- layer-wise probing;
- CKA/representation geometry;
- nuisance probes;
- clinical incremental value;
- FMCIB;
- discrete-time survival sensitivity analysis.

---

# 25. Preregistered priority hierarchy

To avoid post-hoc storytelling, claims should be prioritized.

## Primary

1. H1 — differences in prognostic decodability;
2. H2 — differences in prognostic adaptability;
3. H3 — behavior under limited event supervision.

## Secondary

4. H4 — spatial organization;
5. H5 — representation transformation;
6. H6 — clinical incremental value.

## Exploratory

- individual query semantics;
- layer-specific patterns;
- histology-specific effects;
- stage-specific performance;
- alternative survival losses;
- additional foundation models.

---

# 26. Concrete implementation sequence

## Phase A — feasibility

1. Build harmonized LUNG1 table with patient ID, CT, GTV, OS time and event.
2. Run official preprocessing for all four FMs.
3. Verify successful embedding extraction for at least 95% of eligible subjects.
4. Save global and token embeddings.
5. Run Linear Cox baseline.

**Go/no-go criterion:** all four models can be evaluated on a sufficiently large common cohort.

---

## Phase B — representation audit

1. global Cox probes;
2. tumor-volume/stage probes;
3. intermediate-layer probes;
4. regional pooled probes.

Deliverable: frozen representation atlas.

---

## Phase C — survival adaptation

1. implement SQA;
2. train with frozen backbone;
3. benchmark against MLP Cox;
4. run LoRA/partial FT where feasible.

Deliverable: decodability-adaptability map.

---

## Phase D — event efficiency

1. create fixed nested patient subsets;
2. repeat all core adaptation experiments;
3. report event counts and uncertainty.

Deliverable: survival learning curves.

---

## Phase E — spatial and geometric analysis

1. region-token mapping;
2. regional ablation;
3. CKA / latent-shift analysis;
4. attention/query interpretation.

Deliverable: mechanistic representation analysis.

---

## Phase F — external validation

1. freeze protocol;
2. preprocess NSCLC-Radiogenomics;
3. evaluate locked representations/risk models;
4. test clinical incremental value.

Deliverable: generalization analysis.

---

# 27. Required reproducibility artifacts

The project should release, where licenses permit:

- cohort eligibility code;
- exact patient splits;
- preprocessing configuration per FM;
- cached patient-level embeddings or embedding-generation scripts;
- SQA implementation;
- Cox/discrete-time loss implementation;
- statistical evaluation scripts;
- bootstrap seeds;
- environment files;
- experiment configuration files;
- full hyperparameter search space;
- excluded-patient report.

---

# 28. Literature map supporting the identified gap

| Work | Lung CT | FM representation | True time-to-event | Adaptation | Multiple FM paradigms | Representation-property audit |
|---|---:|---:|---:|---:|---:|---:|
| Pai et al., 2024 | yes | yes | mainly 2-y OS classification | frozen / FT | no | partial |
| LCTfound, 2025 | yes | yes | prognosis task | frozen / FT | no | no |
| CT-CLIP survival, 2026 | yes | yes | yes | frozen / LoRA / FT | no | no |
| ProtoSurv, 2026 | yes | yes | prognosis-oriented; includes survival modeling | prototype-guided | no | no |
| PRISM2, 2026 | pathology | yes | yes | survival cross-attention | no CT comparison | partial |
| **Proposed study** | **yes** | **yes** | **yes** | **common SQA + PEFT** | **yes** | **yes** |

The gap is therefore not the existence of prognostic foundation representations. It is the absence of a **controlled representation-centric benchmark of prognostic transferability across heterogeneous chest CT foundation models**.

---

# 29. Core bibliography

1. **Pai S, Bontempi D, Hadzic I, et al.** Foundation model for cancer imaging biomarkers. *Nature Machine Intelligence.* 2024;6:354–367.  
   https://doi.org/10.1038/s42256-024-00807-9

2. **Allgöwer S, Johansson M, Hallqvist A, et al.** CT-CLIP Representations for Multimodal Lung Cancer Survival Prediction. *MICCAI 2026 Workshops and Challenges / TIA.* 2026.  
   https://arxiv.org/abs/2607.08503  
   https://papers.miccai.org/miccai-2026-sat/TIA_007.html

3. **Liu C, Li S, Sang H, et al.** ProtoSurv: prototype-guided adaptation of computed tomography foundation model for lung-cancer prognosis prediction. *Visual Computing for Industry, Biomedicine, and Art.* 2026;9:21.  
   https://doi.org/10.1186/s42492-026-00233-1

4. **Gao Z, Zhang G, Liang H, et al.** A lung CT vision foundation model facilitating disease diagnosis and medical imaging. *Nature Communications.* 2025.  
   https://doi.org/10.1038/s41467-025-66620-z

5. **TANGERINE authors.** A computationally frugal, open-source chest CT foundation model for thoracic disease detection in lung cancer screening programmes. *Communications Medicine.* 2025.  
   https://doi.org/10.1038/s43856-025-01328-1

6. **Trinh Q-H, Nguyen M-V, Bagci U.** Rad-JEPA 3D: Radiology Joint-Embedding Predictive Model for 3D Computed Tomography. 2026.  
   https://arxiv.org/abs/2607.26196

7. **Microsoft Research.** COLIPRI model and documentation.  
   https://huggingface.co/microsoft/colipri

8. **PRISM2.** End-to-end multimodal pathology foundation model with clinical dialogue. *Nature Medicine.* 2026.  
   https://doi.org/10.1038/s41591-026-04521-4

9. **Bakr S, Gevaert O, Echegaray S, et al.** A radiogenomic dataset of non-small cell lung cancer. *Scientific Data.* 2018;5:180202.  
   https://doi.org/10.1038/sdata.2018.202

10. **NSCLC-Radiomics / LUNG1 dataset.** The Cancer Imaging Archive.  
    DOI: 10.7937/K9/TCIA.2015.PF0M9REI  
    https://www.cancerimagingarchive.net/collection/nsclc-radiomics/

11. **NSCLC-Radiogenomics dataset.** The Cancer Imaging Archive.  
    DOI: 10.7937/K9/TCIA.2017.7hs46erv  
    https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/

---

# 30. One-paragraph proposal abstract

Foundation models pretrained on large-scale chest CT datasets using masked reconstruction, vision-language alignment, joint-embedding prediction and multi-objective learning provide powerful general-purpose representations, but their suitability for prognostic modeling remains insufficiently characterized. Existing studies demonstrate that individual CT foundation models can support lung-cancer survival prediction and can be adapted using frozen heads, parameter-efficient tuning or prognosis-oriented learning; however, it remains unclear whether heterogeneous foundation representations differ systematically in the amount, accessibility, spatial organization and adaptability of survival-relevant information. We propose a controlled benchmark of modern volumetric chest CT foundation models for overall-survival prediction in NSCLC. We will quantify frozen prognostic decodability, survival-specific adaptation gain, event efficiency, tumor/peritumor/global information distribution, representation shift and incremental clinical value using a common censoring-aware evaluation framework. A lightweight Survival Query Adapter will serve as a standardized token-level adaptation probe across frozen encoders. The primary cohort will be NSCLC-Radiomics/LUNG1, with external validation on NSCLC-Radiogenomics. Rather than asking only which foundation model achieves the highest C-index, the study aims to determine **what properties make a general-purpose CT representation prognostically transferable and how much transformation is required for it to become survival-specific**.

---

# 31. Final study thesis

The central thesis of the project is:

> **The prognostic value of a chest CT foundation model is not a single downstream performance number. It is a combination of how much survival information is already encoded, how accessible that information is, how efficiently the representation can be reorganized with limited event supervision, where the information is spatially represented, and whether the resulting survival-specific signal remains robust beyond clinical and technical correlates.**

This thesis is directly testable through the proposed experiments and differentiates the study from existing work that applies one foundation model to one prognostic task or evaluates one adaptation strategy in isolation.
