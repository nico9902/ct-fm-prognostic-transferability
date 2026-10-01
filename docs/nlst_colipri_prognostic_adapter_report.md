# Adapter COLIPRI supervisionato per la prognosi

Data: 22 settembre 2026. Stato: implementazione e smoke test completati; batteria completa avviata in background. Questo documento non anticipa risultati della batteria completa.

## Obiettivo e confini

Verificare se orientare la rappresentazione verso la prognosi migliori la decodificabilità del rischio rispetto all'embedding COLIPRI originale. Questa fase **non addestra un world model**, non predice delta e non modifica COLIPRI. L'eventuale forecasting verrà considerato soltanto dopo l'analisi dei risultati.

Gli esperimenti precedenti suggerivano due problemi distinti: gli adapter congiuntamente addestrati al forecasting perdevano informazione prognostica; gli adapter con ricostruzione e training separato la conservavano, ma non miglioravano chiaramente la predizione temporale. Per questo verifichiamo prima la rappresentazione corrente.

## Componenti riutilizzate e modifiche

Non sono stati modificati i runner precedenti. Riutilizzati:

- `build_pairs`: embedding congelati, etichette post-t+1, diagnostica e split storici.
- `TemporalLatentAdapter`: blocco Linear–LayerNorm–GELU–Linear.
- `fit_stage`: PyTorch Lightning, checkpoint, early stopping, CSVLogger e WandbLogger.
- `fit_probe`: probe lineare GPU con scaling train-only, cinque seed e metriche storiche.
- Funzioni per metriche, seed, stato del processo e hash di provenienza.

Nuovi file:

- `src/lung_jepa/models/prognostic_adapter.py`: modelli e BCE multi-endpoint mascherata.
- `scripts/run_colipri_prognostic_adapter.py`: audit, training, esportazione ordinata, probe, bootstrap e report automatico.
- `configs/model/colipri_prognostic_adapter.yaml`: batteria predefinita.
- `scripts/launch_colipri_prognostic_adapter.sh`: background persistente con GPU UUID, PID e guardia contro doppi lanci.
- `tests/test_prognostic_adapter.py`: controlli CPU.

## Architettura

La standardizzazione delle feature viene stimata solo sulle scansioni **correnti uniche dei pazienti di training**. Non usa feature future, validation o test. Questo è esplicito perché la batteria precedente stimava lo scaler sulle scansioni dei pazienti di training più in generale.

```text
embedding COLIPRI congelato h_t [864]
                  |
        standardizzazione train-only
                  |
                x [864] ----------------------------------+
                  |                                       |
     Linear(864,256) -> LayerNorm -> GELU                   |
                  |                                       |
           Linear(256,864) -> ×0.1 ------------------------+
                                                somma -> z [864]
                                                          |
                                 +------------------------+------------------+
                                 |                                           |
                           Linear(864,3)                    Linear(864,512) -> GELU
                     logits prognostici 1/2/3y                   -> Linear(512,864)
                                                                         ricostruzione x
```

Ultimo layer del ramo residuale inizializzato a zero: z=x all'inizio. Il passaggio interno a 256 dimensioni riguarda la correzione, non l'intero percorso informativo. Il decoder vede solo z, senza bypass da x. Nessun SIGReg, nessun encoder COLIPRI in memoria o in ottimizzazione.

Loss: `lambda_prog * BCE_masked + lambda_rec * MSE(decoder(z),x) + lambda_anchor * MSE(z,x)`.

La BCE applica pesi positivi `negativi/positivi` per ciascun endpoint, calcolati solo sul training. Ogni orizzonte contribuisce in modo uguale alla media; etichette mancanti escluse. La loss di validation è calcolata su tutti gli esempi validi, senza distorsioni dovute alle dimensioni dei minibatch. Non sono imposti vincoli di monotonicità tra i tre output. Le probabilità di una BCE pesata non sono automaticamente rischi calibrati.

## Batteria e selezione

| Variante | Rappresentazione | Peso ricostruzione | Peso anchor | Peso prognosi |
|---|---|---:|---:|---:|
| linear | embedding standardizzato originale | 0 | 0 | 1 |
| mlp | 864→256→864, senza skip | 0 | 0 | 1 |
| adapter_supervised | residuale 864-d | 0 | 0 | 1 |
| adapter_joint_p01 | residuale 864-d | 1 | 0,1 | 0,1 |
| adapter_joint_p1 | residuale 864-d | 1 | 0,1 | 1 |
| adapter_joint_p10 | residuale 864-d | 1 | 0,1 | 10 |

L'MLP ha lo stesso numero di parametri del ramo encoder dell'adapter e la stessa testa; il decoder aggiunge capacità soltanto durante il training dell'obiettivo ausiliario. Il controllo adapter_supervised distingue l'effetto della connessione residuale dall'effetto delle loss di conservazione. La combinazione ricostruzione+anchor è testata come pacchetto: questa batteria non isola causalmente ciascuna delle due componenti.

- Cinque seed di training: 0–4, totale **30 training**.
- AdamW, lr 0,001, weight decay 0,0001, batch 128, massimo 150 epoche, patience 20.
- Checkpoint scelto sulla **BCE prognostica di validation**, identica come criterio tra varianti; non sulla loss totale, i cui pesi differiscono.
- Tra i tre adapter joint, selezione sulla media della migliore BCE validation nei cinque seed, prima della valutazione con i nuovi probe. Nessuna scelta basata sul test.
- Ogni modello usa soltanto lo stato corrente: niente round, delta-t, storia o feature future negli input.

## Endpoint e controlli sulla coorte

Identiche 1.984 coppie / 998 pazienti della batteria precedente. Asserzioni controllano corrispondenza di paziente, coppia, indici, split ed etichette. Split per paziente, mai per coppia. Un paziente può contribuire due righe allo stesso split; il training resta per coppia, come il protocollo storico.

Etichette: prima diagnosi di cancro entro 365/730/1095 giorni **dopo t+1**. Diagnosi on/before t+1 e follow-up insufficiente possono produrre etichette mancanti; non vengono convertite in negativi. Non si modifica l'endpoint né si esclude un nuovo sottogruppo.

| Endpoint | Train positivi/validi | Validation | Test |
|---|---:|---:|---:|
| 1y | 107/1186 | 33/394 | 31/396 |
| 2y | 256/1181 | 81/391 | 75/396 |
| 3y | 364/1173 | 112/389 | 113/396 |

Il problema 1y è fortemente diverso tra round: train T0→T1 5/596 e T1→T2 102/590; test rispettivamente 2/199 e 29/197. Viene quindi incluso un controllo **solo round**, con probabilità stimate dalla prevalenza di ciascun endpoint/round nel training. È valutato sugli stessi pazienti di tutti gli altri modelli. Non viene usato come input alle reti.

La documentazione della coorte conferma un campionamento deliberato 500 ever-cancer / 500 never-cancer: non è la prevalenza naturale NLST. Disponibilità delle scansioni e criteri storici di selezione possono influire sullo sbilanciamento temporale, ma l'audit presente non identifica causalmente il contributo di ciascun criterio. Non si conclude che le etichette siano sbagliate. Il controllo per round e le analisi stratificate sono necessari ma non eliminano ogni bias di selezione.

Esportati `class_distribution.csv`, `diagnosis_lags.csv`, `round_probabilities.npz` e manifest delle coppie. Questo audit è descrittivo; non modifica la coorte dopo aver visto il test.

## Valutazione della rappresentazione

Due analisi separate:

1. **Testa addestrata congiuntamente**, cinque seed per variante: misura il sistema supervisionato completo.
2. **Nuovo probe lineare sul modello congelato**, cinque seed indipendenti per ogni seed dell'encoder: verifica quanto l'informazione sia accessibile anche a un nuovo classificatore.

Il probe congelato usa lo stesso runner storico: scaling training-only, Linear+BCE pesata, AdamW 0,001/0,0001, batch 128, massimo 100 epoche, early stopping 15. La variante linear ha identica rappresentazione all'input standardizzato: il probe nativo la copre senza duplicazioni.

Riferimenti: probe su embedding COLIPRI corrente e su embedding futuro reale. Quest'ultimo è solo oracle: nessuna feature futura entra nel training dell'adapter. Totale **405 fit di probe**: `(25 encoder addestrati + 2 riferimenti) × 3 endpoint × 5 seed`.

Metriche AUROC, AUPRC, MCC e F1; soglia 0,5 come nel protocollo precedente, non ottimizzata sul test. Report pooled e per T0→T1/T1→T2, usando modelli addestrati sulle coppie pooled, non riaddestrati per sottogruppo. Predizioni ed embedding non vengono mediati nella valutazione primaria.

Analisi secondaria: media delle probabilità delle teste sui seed, bootstrap appaiato per paziente (1000 ricampionamenti), adapter joint selezionato contro linear e mlp. Si mantengono insieme le coppie dello stesso paziente e si riportano i ricampionamenti validi; campioni con una sola classe sono scartati. Gli intervalli descrivono l'incertezza sul campione test fisso, non l'incertezza della selezione del modello. Con due positivi T0→T1, l'endpoint 1y rimane esplorativo.

Il test è già stato esaminato negli esperimenti precedenti: non è una nuova validazione indipendente. Prima di affermazioni cliniche servirebbe una coorte non utilizzata nello sviluppo. Inoltre, l'eventuale sovrapposizione con il pretraining COLIPRI non è risolta da questa batteria.

## Artefatti e monitoraggio

Directory: `outputs/colipri_prognostic_adapter/`.

- `config.json`, `fingerprint.json`, `provenance.json`: configurazione, hash di codice/input, GPU e PID. Resume rifiutato se cambiano gli input o il codice fingerprintato.
- `scalers.npz`: scaler e pesi di classe training-only.
- `runs/<variante>__seedN/best.ckpt`, `history/`, `done.json`: checkpoint, training e URL W&B.
- `representations.npz`: stato corrente trasformato, logits, probabilità e identificatori ordinati delle coppie.
- `representation_diagnostics.json`: loss prognostica per split, ricostruzione, variazione dallo stato originale e dispersione delle feature.
- `probes/`: checkpoint e scaler dei nuovi probe, probabilità ed etichette con pair IDs.
- `validation_selection.csv`, `selection.json`: scelta dell'adapter esclusivamente su validation.
- `metrics.csv`, `summary.csv`: metriche per seed e aggregate.
- `secondary_ensemble_patient_bootstrap.csv`: analisi secondaria delle differenze.
- `status.json`, `run.log`, `COMPLETE.json`: avanzamento e completamento.
- `RESULTS.md`: **report dei risultati generato automaticamente al completamento**, con validation/test e tutte le transizioni. Non esiste ancora un risultato completo al momento del lancio.

Tutti i training e i probe GPU, incluso lo smoke test, vengono eseguiti in background e registrati su W&B, progetto `nlst-colipri-temporal-adapter`, gruppo `prognostic-adapter-v1` (smoke separato).

```bash
bash scripts/launch_colipri_prognostic_adapter.sh GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01 --smoke
bash scripts/launch_colipri_prognostic_adapter.sh GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01
```

## Verifiche e stato operativo

Quattro test CPU superati: mascheramento delle etichette e gradienti, batch interamente senza etichette, identità iniziale e gradiente prognostico sull'encoder, equivalenza dello stato ricaricato per tutte le varianti. `pytest` non è installato nell'ambiente: le quattro funzioni sono state eseguite direttamente, senza installare dipendenze. Compilazione Python e controllo sintattico Bash superati.

`nvidia-smi` iniziale: GPU 3 A100 80GB libera, GPU 0 occupata da un processo non toccato. Smoke test eseguito in background, PID 3212291, con tre modelli / un seed / due epoche e dodici probe: **completato alle 11:40:33 CEST**. Tutti gli embedding esportati sono finiti e di forma `[1984,864]`; identificatori delle coppie e label dei dodici probe coincidono esattamente con i manifest. W&B sincronizzato: [smoke evaluation](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/5wgkdm6e).

Batteria completa avviata il 22 settembre, PID **3215618**, GPU fisica **3 / A100 80GB**, UUID `GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01`. Verifica successiva con `nvidia-smi`: processo presente sulla GPU corretta, circa 622 MiB allocati. Snapshot delle 11:45:48 CEST: training `linear`, seed 1; il primo seed è già completato. W&B online confermato: [linear seed 1](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/856t619j).

Per lo stato successivo a questo snapshot usare `outputs/colipri_prognostic_adapter/status.json`. Il report completo verrà generato in `outputs/colipri_prognostic_adapter/RESULTS.md` e il completamento segnalato da `COMPLETE.json`. Nessun risultato dello smoke viene interpretato come risultato scientifico.
