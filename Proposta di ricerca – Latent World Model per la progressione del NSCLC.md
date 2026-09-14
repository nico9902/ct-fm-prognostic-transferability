# Latent World Model for Longitudinal NSCLC Progression

## 1. Obiettivo e domanda di ricerca

L'obiettivo del progetto è sviluppare un **world model multimodale per il carcinoma polmonare non a piccole cellule (NSCLC)** capace di apprendere come lo stato latente del paziente evolve nel tempo a partire da immagini CT longitudinali, informazioni cliniche e, quando disponibili, trattamenti ricevuti.

La domanda di ricerca principale è:

**L'apprendimento esplicito delle dinamiche temporali della malattia consente di ottenere rappresentazioni del paziente più informative per la prognosi e la risposta al trattamento rispetto ai modelli statici o ai normali modelli longitudinali discriminativi?**

Formalmente, invece di apprendere solamente:

\[
CT_t + Clinical_t \rightarrow Outcome
\]

il modello deve apprendere:

\[
z_t,\,a_t,\,\Delta t \rightarrow \hat z_{t+1}
\]

dove \(z_t\) rappresenta lo stato latente del paziente, \(a_t\) l'eventuale trattamento e \(\Delta t\) l'intervallo temporale.

---

## 2. Ipotesi da dimostrare

L'ipotesi centrale è che:

> **un modello obbligato a comprendere come il tumore e il paziente evolvono nel tempo apprenda una rappresentazione clinicamente più significativa rispetto a un encoder addestrato esclusivamente per predire un endpoint.**

Il progetto dovrà dimostrare tre punti principali.

**H1 – Future-state prediction.**  
Il modello deve riuscire a predire lo stato latente osservato alla visita successiva:

\[
\hat z_{t+1}\approx z_{t+1}.
\]

**H2 – Prognostic representation learning.**  
Le rappresentazioni apprese attraverso la previsione della progressione devono migliorare la predizione di endpoint come:

- Overall Survival;
- Progression-Free Survival, quando disponibile;
- risposta al trattamento;
- progressione/stadio;
- variazione del tumor burden.

**H3 – Long-term dynamics.**  
Il modello deve mantenere informazione clinicamente significativa anche durante rollout di più step:

\[
z_t\rightarrow\hat z_{t+1}\rightarrow\hat z_{t+2}.
\]

L'obiettivo non è quindi semplicemente generare una CT futura, ma verificare se **comprendere la dinamica della malattia migliora la capacità di rappresentare e prognosticare il paziente**.

---

## 3. Competitor e baseline da battere

Il principale riferimento metodologico è **Medical World Model (MeWM, ICCV 2025)**, che simula l'evoluzione tumorale condizionata dal trattamento tramite modelli generativi. La nostra proposta si differenzia concentrandosi principalmente sulla previsione dello stato futuro nello spazio latente, evitando la necessità di ricostruire immagini realistiche.

Il competitor più diretto in ambito polmonare è però il progetto NLST 2026 **Treatment-Conditioned World Models for Lung Cancer Progression Prediction**, che utilizza un encoder 3D Merlin e un **Recurrent State Space Model derivato da DreamerV3** per prevedere longitudinalmente embedding CT condizionati dai trattamenti. Il progetto prevede anche survival prediction e analisi controfattuale.

Di conseguenza, non sarebbe sufficiente proporre semplicemente un RSSM applicato a NLST.

Le baseline sperimentali dovrebbero comprendere:

1. **Static model**
\[
CT_t\rightarrow Outcome
\]

2. **Static multimodal model**
\[
CT_t+Clinical_t\rightarrow Outcome
\]

3. **Longitudinal discriminative model**
\[
CT_t+CT_{t+1}\rightarrow Outcome
\]

4. **Feature-difference model**
\[
z_{t+1}-z_t\rightarrow Outcome
\]

5. **RSSM/Dreamer-style latent world model**, simile al progetto NLST già proposto.

6. **Proposed JEPA-style latent world model**.

L'obiettivo principale dovrebbe quindi essere mostrare:

\[
WorldModel_{JEPA}
>
RSSM
>
Temporal/Static\ Baselines
\]

sulla qualità della previsione futura e soprattutto sulla capacità prognostica delle rappresentazioni ottenute.

---

## 4. Dataset

### NLST – dataset principale

Il **National Lung Screening Trial** rappresenta la risorsa principale per il progetto grazie alla presenza di grandi quantità di CT longitudinali e follow-up clinico.

Il progetto NLST-1502 attualmente approvato stima circa:

- 75.000 CT nei round longitudinali;
- circa 26.000 partecipanti nel CT arm;
- circa 1.900 pazienti con carcinoma polmonare trattato;
- circa 4.600 procedure terapeutiche;
- informazioni su chirurgia, radioterapia e chemioterapia;
- follow-up di sopravvivenza fino a 12 anni.

NLST potrebbe quindi essere utilizzato inizialmente per apprendere:

\[
CT_t\rightarrow CT_{t+1}
\]

e successivamente:

\[
CT_t+Treatment_t\rightarrow CT_{t+1}.
\]

### 4D-Lung – validazione treatment-conditioned

TCIA 4D-Lung contiene CT longitudinali di **20 pazienti con NSCLC localmente avanzato durante chemioradioterapia**, con acquisizioni 4D-FBCT e 4D-CBCT ripetute durante il trattamento.

Il campione è troppo piccolo per il training principale, ma è particolarmente interessante per verificare se lo spazio latente rappresenta realmente l'evoluzione tumorale durante il trattamento.

### LUNG1 / NSCLC-Radiomics – transfer prognostico

LUNG1 contiene **422 pazienti NSCLC**, CT pre-trattamento, segmentazioni tumorali e informazioni di follow-up.

Non essendo longitudinale, non può essere utilizzato per addestrare il world model. Può però rispondere a una domanda molto importante:

> Una rappresentazione appresa attraverso la previsione dell'evoluzione della malattia è più utile per la prognosi rispetto a una normale rappresentazione CT?

Si può quindi trasferire l'encoder ottenuto su NLST e confrontarlo con encoder ImageNet, self-supervised o foundation model.

---

## 5. Metodo proposto

La proposta principale è un **JEPA-style Latent Medical World Model**.

### Encoder dello stato corrente

La CT viene codificata da un encoder 3D:

\[
CT_t\xrightarrow{E_\theta} z^{CT}_t.
\]

Le informazioni cliniche vengono codificate separatamente:

\[
Clinical_t\xrightarrow{E_c}z^{clinical}_t.
\]

Le rappresentazioni vengono fuse:

\[
z_t=Fusion(z^{CT}_t,z^{clinical}_t).
\]

Quando disponibili vengono aggiunti:

\[
Treatment_t,\quad \Delta t.
\]

### Predictive World Model

Un Transformer/State-Space Predictor riceve:

\[
(z_t,a_t,\Delta t)
\]

e predice:

\[
\hat z_{t+1}.
\]

La CT realmente acquisita al tempo successivo viene invece elaborata attraverso un **target encoder/EMA teacher**:

\[
CT_{t+1}\rightarrow z_{t+1}.
\]

Il training avviene imponendo:

\[
\mathcal L_{dyn}
=
D(\hat z_{t+1},sg(z_{t+1})),
\]

dove \(sg\) indica lo stop-gradient.

Il modello quindi **non deve ricostruire i voxel della CT futura**, ma prevedere la rappresentazione clinicamente significativa del suo stato futuro.

### Obiettivi clinici

Alla rappresentazione possono essere aggiunte teste dedicate:

\[
z_t,\hat z_{t+1}
\rightarrow Survival
\]

\[
z_t,\hat z_{t+1}
\rightarrow Treatment\ Response
\]

\[
z_t,\hat z_{t+1}
\rightarrow Tumor\ Burden.
\]

Il training complessivo potrebbe essere:

\[
\mathcal L=
\lambda_{dyn}\mathcal L_{dyn}
+\lambda_{surv}\mathcal L_{survival}
+\lambda_{resp}\mathcal L_{response}.
\]

---

## 6. Contributo originale atteso

Il contributo non dovrebbe essere presentato semplicemente come *“un world model per il tumore al polmone”*, perché un progetto simile basato su RSSM è già stato approvato su NLST nel 2026.

Il gap più interessante è invece:

> **studiare se un paradigma JEPA-like di predictive representation learning possa apprendere dinamiche longitudinali del NSCLC più clinicamente informative rispetto a modelli statici, longitudinali discriminativi e world model RSSM/Dreamer-style.**

La possibile novità è quindi l'unione di:

**longitudinal CT + multimodal patient state + latent predictive learning + world modeling + prognosis.**

Il risultato centrale da ottenere sarebbe dimostrare che:

\[
\boxed{
\text{Learning disease dynamics produces better prognostic representations}
}
\]

ovvero che imparare **come il paziente evolverà** consente di rappresentarne meglio anche lo stato corrente e di predirne più accuratamente l'outcome.

In una seconda fase, il framework potrebbe essere esteso alla simulazione condizionata dal trattamento e, con appropriate tecniche causali, alla valutazione controfattuale di strategie terapeutiche.