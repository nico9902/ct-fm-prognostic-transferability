# Evidenze per la conclusione: il limite sta nell'input, non nel modello

Data: 2026-10-29. Ricerca mirata (PubMed, PMC, web), non revisione sistematica.
Scopo: dare una base bibliografica verificata alla chiusura dell'abstract e alle sezioni Discussion, Limitations e Conclusions.

Legenda verifica:
- **FT** = full text letto (PMC via PubMed);
- **ABS** = solo abstract PubMed letto;
- **2a mano** = dato riportato da un altro articolo, non controllato sulla fonte primaria.

---

## 1. Chiusura dell'abstract adottata

Inserita in `main.tex`; sostituisce la frase finale precedente.

> Across clinical, radiomic, tumour-centred and foundation models, and across every level of adaptation, discrimination converged near 0.6, and the foundation models drew on the patient rather than on the tumour. With pretreatment CT and the clinical variables that public cohorts provide, the limit to survival prediction thus appears to lie in the input rather than in the model: further progress will require information such as performance status, comorbidity and treatment, which these cohorts do not record, rather than larger or more extensively adapted image encoders.

Frase precedente (rimossa, per riferimento):

> Frozen foundation-model embeddings of the whole CT thus carried prognostic information comparable to, but not demonstrably greater than, established predictors, and depended little on the tumour; further adaptation did not help, and absolute risks require recalibration before external use.

Nota: il risultato sulla calibrazione esterna (rischi predetti dal 37% all'85% contro il 25% osservato) resta nel corpo dell'abstract.

---

## 2. L'argomento in breve

1. **I nostri dati: convergenza.** Tutte le famiglie di modelli si concentrano tra C ≈ 0.59 e 0.63:
   - Stage+Volume 0.588, radiomica 0.591, COLIPRI 0.617 (0.634 con PCA32);
   - nessun livello di adattamento migliora la readout lineare (tutti i p_Holm ≥ 0.41);
   - raddoppiare la capacità della readout (PCA32) non cambia nulla;
   - gli encoder non sono più efficienti dei baseline al diminuire dei pazienti di training;
   - mascherare il tumore lascia la discriminazione equivalente (margine ±0.03);
   - il probing mostra che gli embedding codificano età e sesso più del volume tumorale (tranne COLIPRI).
2. **Letteratura: mancano variabili importanti.** I determinanti clinici della sopravvivenza in NSCLC trattato con (chemio)radioterapia comprendono variabili che LUNG1 non contiene: WHO performance status, trattamento (regime, dose, durata), FEV1, stazioni linfonodali positive alla PET e biomarcatori ematici (§3).
3. **Letteratura: anche con quelle variabili il tetto è basso.** Sulla stessa istituzione (MAASTRO), un modello con tutte quelle variabili ottiene C = 0.62 corretto per l'ottimismo, e 0.58–0.60 in validazione esterna (§3.1). È lo stesso livello dei nostri modelli basati sulla sola CT.
4. **Letteratura: su LUNG1 il tetto è noto.** Su LUNG1 il plateau è già documentato per radiomica, deep feature 2D e CNN. Gli autori di quello studio indicavano come sviluppi futuri modelli 3D, apprendimento contrastivo e analisi dell'intero polmone (§3.4). Questo lavoro fa esattamente quello, e il plateau resta.
5. **Conclusione difendibile.** Con la CT pretrattamento e le variabili cliniche pubbliche, il limite *sembra* stare nell'input. Per andare oltre servono dati sull'ospite e sul trattamento, non encoder più grandi. Non è difendibile dire "il task non si può fare", perché in clinica esistono predittori che non abbiamo usato.

---

## 3. Fonti verificate

### 3.1 Oberije et al., 2015 — fonte chiave **[FT]**

Oberije C, De Ruysscher D, Houben R, et al. *A Validated Prediction Model for Overall Survival From Stage III Non-Small Cell Lung Cancer: Toward Survival Prediction for Individual Patients.* Int J Radiat Oncol Biol Phys. 2015;92(4):935–944.
DOI: [10.1016/j.ijrobp.2015.02.048](https://doi.org/10.1016/j.ijrobp.2015.02.048) · PMID 25936599 · PMC4786012

**Coorte**
- 548 pazienti con NSCLC in stadio III inoperabile (TNM 6ª ed.), trattati con RT a intento curativo **alla MAASTRO Clinic**, marzo 2002 – agosto 2011.
- Dati raccolti prospetticamente; stadiazione con PET obbligatoria.
- Endpoint: OS dall'inizio della RT. Il 83.2% dei pazienti era deceduto, con follow-up mediano di 5.5 anni.

**Variabili candidate** (verbatim): *"treatment cohort (stratification factor), age, gender, WHO performance status, body mass index, forced expiratory volume (in 1 second), clinical overall stage, T stage, N stage, number of positive lymph node stations (PLNS), GTV, OTT, and EQD2"*.

**Modello finale**
- Il testo dei Risultati riporta: gender, WHO performance status, **T stage**, GTV, PLNS, OTT ed EQD2, più l'età inclusa anche se non significativa. Il modello è stratificato per coorte di trattamento.
- ⚠ L'abstract elenca invece *age, gender, WHO PS, OTT, EQD2, PLNS, GTV* e **omette il T stage**. Se citiamo le variabili, conviene usare quelle dei Risultati.

**Discriminazione**
- *"The c statistic of the final model was 0.65 (0.62 after correction for optimism)"*: il C è di Harrell, con correzione bootstrap.
- Esterno: *"c statistics of 0.58 and 0.60"* rispettivamente per NKI (n=174) e MSKCC (n=130).
- ⚠ La Discussione dello stesso articolo riporta invece "0.59 and 0.60". Conviene citare i valori dei Risultati (0.58 e 0.60) o scrivere "0.58–0.60".
- Modelli basati solo sul TNM nella stessa coorte: stadio complessivo C = 0.50; T+N C = 0.55. Con la stratificazione per trattamento salgono a 0.54 e 0.57.

**Citazioni utili**
- *"The performance of the model in the development cohort and in the validation data sets was moderate, but evidently better than the performance of the TNM-based models."*
- *"performance status, regardless of resectability of the tumor, is consistently identified as an important predictor for survival"*
- *"The GTV was the most important prognostic factor in our analysis."*
- *"Although new factors are needed to improve the predictive capability of the survival model …"*
- *"integration of information from multiple sources is the way forward toward more accurate prediction models. For that purpose, large databases, new modeling methods, and a culture of data sharing are needed"*
- Sul confondimento per indicazione: *"the choice of treatment regimen was partially guided by patients' general health and physical fitness"*. È rilevante per noi, perché LUNG1 non registra il trattamento.

**Dati**: il dataset è pubblico su cancerdata.org (https://www.cancerdata.org/10.1016/j.ijrobp.2015.02.048).

**Come usarla**
- Ancoraggio: anche con PS, trattamento, dose, FEV1/BMI tra le candidate, PLNS e GTV, nella stessa istituzione da cui proviene LUNG1, il C resta 0.62 internamente e 0.58–0.60 esternamente.
- Lo stadio da solo vale circa 0.50–0.55, coerente con il nostro modello Clinical a 0.529.

**Cautele**
- La popolazione comprende solo pazienti in stadio III (LUNG1 include stadi I–IIIb).
- C di Harrell contro il nostro Uno a 36 mesi.
- Nessun confronto diretto: non possiamo affermare "uguale a".
- La sovrapposizione di pazienti con LUNG1 è plausibile (stesso centro, periodo compatibile) ma **non verificata**.

### 3.2 Dehing-Oberije et al., 2009 **[ABS]**

Dehing-Oberije C, Yu S, De Ruysscher D, et al. *Development and external validation of prognostic model for 2-year survival of non-small-cell lung cancer patients treated with chemoradiotherapy.* Int J Radiat Oncol Biol Phys. 2009;74(2):355–362.
DOI: [10.1016/j.ijrobp.2008.08.052](https://doi.org/10.1016/j.ijrobp.2008.08.052) · PMID 19095367

- 377 pazienti NSCLC inoperabili, stadio I–IIIB, trattati con chemioradioterapia (MAASTRO), più due dataset esterni.
- Modello (SVM): gender, WHO performance status, FEV1, number of positive lymph node stations, GTV.
- Endpoint: sopravvivenza binaria a 2 anni. AUC 0.74 in LOO-CV; 0.75 e 0.76 sui due set esterni.
- L'abstract dice anche: *"the clinical TNM stage is highly inaccurate for the prediction of survival, and alternatives are lacking."*
- **Cautela**: è un'AUC binaria a 2 anni, non un C-index di sopravvivenza, quindi non è confrontabile numericamente con i nostri valori. Serve solo per mostrare *quali* variabili contano (PS, FEV1, PLNS).

### 3.3 Dehing-Oberije et al., 2010 **[ABS]**

Dehing-Oberije C, Aerts H, Yu S, et al. *Development and validation of a prognostic model using blood biomarker information for prediction of survival of non-small-cell lung cancer patients treated with combined chemotherapy and radiation or radiotherapy alone.* Int J Radiat Oncol Biol Phys. 2010;81(2):360–368.
DOI: [10.1016/j.ijrobp.2010.06.011](https://doi.org/10.1016/j.ijrobp.2010.06.011) · PMID 20888135

- Coorte prospettica: 106 pazienti, stadio I–IIIB, inoperabili, RT o chemio-RT; validazione su n=52.
- Modello base (gender, WHO PS, FEV, PLNS, GTV): AUC LOO 0.72 → modello esteso con **CEA e IL-6**: 0.81 (p=0.004).
- In validazione: 0.66 → 0.76.
- Uso: i biomarcatori ematici aggiungono informazione oltre la clinica e l'imaging. Esempio concreto di "informazione assente dall'input" che migliora la predizione.
- **Cautela**: campione piccolo, AUC binaria; il numero di Volume/Issue nel testo PubMed è 81(2) del 2010 (pubblicato online 2010, issue 2011). Controllare la citazione finale sul sito dell'editore.

### 3.4 Braghetto et al., 2022 — LUNG1, plateau già noto **[FT]**

Braghetto A, Marturano F, Paiusco M, Baiesi M, Bettinelli A. *Radiomics and deep learning methods for the prediction of 2-year overall survival in LUNG1 dataset.* Sci Rep. 2022;12:14132.
DOI: [10.1038/s41598-022-18085-z](https://doi.org/10.1038/s41598-022-18085-z) · PMID 35986072 · PMC9391464
⚠ Autori, numero di articolo (14132) e lista completa: da ricontrollare sul sito dell'editore; il full text PMC ottenuto non includeva il blocco autori.

**Dataset e variabili**
- LUNG1, 411 dei 422 pazienti.
- Variabili cliniche usate: *"sex, age at diagnosis, TNM staging, AJCC staging and histology"*, cioè le stesse nostre (più M).
- Dichiarano che il dataset pubblico contiene, per paziente, solo *"seven clinical parameters (i.e., sex, age at diagnosis, TNM staging, AJCC staging, histology, survival time, and death status)"*.

**Risultati** (sopravvivenza binaria a 2 anni, AUC sul test, 5 split casuali 3:1)

| Modello | AUC test |
|---|---|
| Solo clinica | ≤ 0.59 |
| Radiomica | 0.67 ± 0.03 |
| Deep feature 2D (CAE) | 0.63 ± 0.03 |
| Radiomica + deep feature | 0.67 ± 0.02 |
| CNN 2D | 0.64 ± 0.04 |

**Citazioni utili**
- Abstract: *"Despite the high number of pipelines and approaches tested, results were comparable and in line with previous works, hence confirming that it is challenging to extract further imaging-based information from the LUNG1 dataset for the prediction of 2-year OS."*
- *"the considered clinical data—available with the LUNG1 dataset—do not significantly contribute to the 2-year OS prediction in NSCLC."*
- *"In future investigations it would also be interesting to test whether other clinical features or prognostic biomarkers (e.g., gene features) may improve survival prediction"*
- Conclusione: *"the image-based prediction of the 2-year OS on the LUNG1 dataset is a challenging task, with limited margins of improvements."*
- **Sviluppi futuri proposti:** *"to apply different DL learning techniques (e.g., contrastive learning), to perform a whole-lung analysis, hence including information from the tissue surrounding the GTV, or to employ three-dimensional DL models"*

**Come usarla**
- Questo è l'argomento più forte: il nostro studio realizza esattamente le direzioni future che Braghetto indicava (encoder 3D, pretraining contrastivo come COLIPRI e CT-CLIP, CT intera) e il plateau resta. Rafforza l'idea che il limite non sia metodologico ma informativo.

**Cautele**
- AUC binaria a 2 anni, non il nostro C di Uno a 36 mesi; numeri non confrontabili direttamente.
- Senza un set esterno.

### 3.5 Woodard, Jones, Jablons, 2016 — fattori prognostici **[ABS]**

Woodard GA, Jones KD, Jablons DM. *Lung Cancer Staging and Prognosis.* Cancer Treat Res. 2016;170:47–75.
DOI: [10.1007/978-3-319-40389-2_3](https://doi.org/10.1007/978-3-319-40389-2_3) · PMID 27535389 (capitolo di review)

- *"TNM stage remains the most important prognostic factor in predicting recurrence rates and survival times, followed by tumor histologic grade, and patient sex, age, and performance status."*
- *"despite intense research efforts and countless publications, no molecular prognostic marker has been adopted into clinical use since most fail in subsequent cross-validation"*
- Uso: fonte generale per l'elenco dei fattori prognostici consolidati, tra cui il performance status, che LUNG1 non ha. Il grading istologico non c'è in LUNG1 ma c'è in NSCLC-Radiogenomics.
- **Cautela**: capitolo di libro, non studio primario. Per il PS sarebbe preferibile aggiungere una fonte primaria IASLC (vedi §5, da fare).

### 3.6 Fonti già nel manoscritto rilevanti per l'argomento (non ri-verificate qui)

- **Aerts et al., 2014** (`aerts2014`, DOI [10.1038/ncomms5006](https://doi.org/10.1038/ncomms5006)): firma radiomica su LUNG1. Braghetto riporta un C di validazione pari a 0.65 per il polmone (2a mano).
- **Hosny et al., 2018** (`hosny2018`): CNN centrata sul tumore, test su un sottoinsieme MAASTRO (n=307). Rimuovere il contesto peritumorale riduceva l'AUC da 0.70 a 0.63 (da `literature_notes.md`).
- **Neukirch et al., 2026** (`neukirch2026`): feature handcrafted pari alle feature dei foundation model su LUNG1.
- **Allgöwer et al., 2026** (`allgower2026`): LoRA e fine-tuning completo di CT-CLIP non migliorano sull'encoder congelato.
- **Pai et al., 2024** (`pai2024`): encoder delle lesioni congelato migliore delle varianti fine-tuned.

### 3.7 Fonti trovate ma sconsigliate

- **Didier et al., 2024**, Front Artif Intell, DOI [10.3389/frai.2024.1365777](https://doi.org/10.3389/frai.2024.1365777) **[ABS]**: meta-analisi di modelli ML contro regressione logistica per l'OS nel tumore polmonare (C 0.60–0.85; media pesata 0.78 per ML contro 0.70 per LR). Popolazioni eterogenee, per lo più registri con variabili cliniche e non imaging. Non è confrontabile e rischia di suggerire ai revisori che C ≈ 0.78 sia raggiungibile. Sconsigliata come supporto.
- **DeepMMSA (2021/2022, arXiv 2106.06744)** e **Lite-ProSENet** (dalla ricerca web): fusione di CT e dati clinici su TCIA (probabilmente LUNG1) con C-index molto alti dichiarati (Lite-ProSENet 0.893). Non verificati. Se un revisore li cita, va controllato il protocollo: possibile leakage o assenza di validazione corretta. Da verificare prima di menzionarli.

---

## 4. Variabili disponibili e mancanti nelle nostre coorti

| Variabile (rilevanza clinica) | LUNG1 | NSCLC-Radiogenomics | Fonte del ruolo prognostico |
|---|---|---|---|
| Stadio, T, N | ✔ | ✔ (patologico) | Woodard 2016; Oberije 2015 |
| Età, sesso | ✔ | ✔ | Woodard 2016; Oberije 2015 |
| Istologia | ✔ | ✔ | — |
| Volume tumorale (GTV) | ✔ (da segmentazione) | ✔ (da segmentazione) | Oberije 2015 (*"most important prognostic factor"*) |
| **WHO/ECOG performance status** | ✘ | ✘ | Oberije 2015; Dehing-Oberije 2009, 2010; Woodard 2016 |
| **Trattamento** (RT sola / sequenziale / concomitante, dose EQD2, OTT) | ✘ | parziale (adjuvant, chemo, radiation: sì/no) | Oberije 2015 |
| **FEV1 / funzione polmonare** | ✘ | ✘ | Dehing-Oberije 2009, 2010 |
| **Stazioni linfonodali positive (PET)** | ✘ | ✘ | Oberije 2015; Dehing-Oberije 2009 |
| **Biomarcatori ematici** (CEA, IL-6, LDH) | ✘ | ✘ | Dehing-Oberije 2010 |
| Grading istologico | ✘ | ✔ | Woodard 2016 |
| Fumo / pack-years | ✘ | ✔ | Oberije 2015 (lo cita come possibile spiegazione delle differenze tra coorti) |
| EGFR / KRAS / ALK | ✘ | ✔ | Woodard 2016 (marker molecolari non adottati come prognostici) |
| Peso | ✘ | ✔ (lbs, senza variazione nel tempo) | — |

Il modello Clinical ha usato solo le variabili comuni alle due coorti ([methods.tex:65](../sections/methods.tex)).

---

## 5. Cose da fare prima di scrivere Discussion e Conclusions

1. **Fonte primaria sul performance status**: cercare l'analisi dei fattori prognostici IASLC (Sculier et al., J Thorac Oncol 2008, sui fattori prognostici aggiuntivi nel database IASLC). La ricerca PubMed per titolo non ha trovato il record: va recuperato con DOI.
2. **Verificare i dati di Braghetto** (autori, numero articolo) sul sito Nature.
3. **Sovrapposizione Oberije 2015 / LUNG1**: confrontare il dataset su cancerdata.org con gli ID di LUNG1. Se collegabile, si potrebbe aggiungere il PS (e il trattamento) a LUNG1, cosa che cambierebbe l'analisi e andrebbe discussa prima.
4. **Analisi esplorativa opzionale** su NSCLC-Radiogenomics con il modello clinico arricchito (fumo, grading, invasione, mutazioni). Con 33 eventi a 36 mesi è sottopotenziata, utile solo come sensitivity analysis.
5. **Aggiungere al `references.bib`**: oberije2015, dehingoberije2009, dehingoberije2010, braghetto2022, woodard2016 (dopo le verifiche sopra).

---

## 6. Bozza di contenuti per le sezioni (da sviluppare)

**Limitations**: un nuovo paragrafo sulle variabili assenti. LUNG1 e NSCLC-Radiogenomics non registrano performance status, trattamento ricevuto, funzione polmonare, stazioni linfonodali alla PET né biomarcatori ematici. Questi sono predittori consolidati nell'NSCLC trattato con (chemio)radioterapia (Oberije 2015; Dehing-Oberije 2009, 2010). Il modello clinico è quindi un modello delle variabili *disponibili*, non dello stato dell'arte clinico.

**Discussion** (comparison with previous work / implications):
- Il plateau a ~0.6 su LUNG1 era già documentato per radiomica e CNN 2D (Braghetto 2022). Le direzioni proposte allora (3D, contrastivo, polmone intero) sono quelle testate qui, e il plateau persiste.
- Nella stessa istituzione, un modello con PS, trattamento, dose, PLNS e GTV arrivava a C 0.62 (0.58–0.60 esterno; Oberije 2015). Con la dovuta cautela sulle differenze di popolazione e di stimatore, la sola CT arriva quindi allo stesso livello di un modello clinico completo.
- Ipotesi (post hoc, da segnalare come tale): gli embedding dell'intera CT codificano soprattutto età, sesso e anatomia del paziente, e potrebbero catturare in parte proxy dello stato generale. Questo spiegherebbe perché il tumore conta poco, ma non è stato testato.

**Conclusions / messaggio alla comunità**:
- Il collo di bottiglia della predizione di sopravvivenza da CT pretrattamento nelle coorti pubbliche è l'informazione disponibile, non la capacità del modello.
- I dataset di imaging pubblici per la prognosi dovrebbero rilasciare performance status, trattamento e comorbidità (in linea con l'appello di Oberije 2015 alla condivisione dei dati).
- Gli studi futuri dovrebbero confrontarsi con un modello clinico completo, non solo con le variabili disponibili.
