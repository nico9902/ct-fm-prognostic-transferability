# Prognosi COLIPRI dalla progressione longitudinale osservata

Data: 22 settembre 2026. Questo documento definisce il protocollo **prima** del training completo. I risultati verranno aggiunti soltanto dopo il completamento della batteria.

## Motivazione

Gli esperimenti precedenti hanno mostrato che:

- il futuro reale COLIPRI è molto più prognostico dello stato corrente, soprattutto a 1 e 2 anni;
- i world model predicono una parte del cambiamento medio, ma recuperano solo una piccola parte del vantaggio prognostico del futuro reale;
- l'adapter residuale 864-d conserva COLIPRI, ma la supervisione prognostica non crea una rappresentazione stabilmente migliore;
- una trasformazione bottleneck non residuale tende invece a perdere informazione;
- nel dataset pooled precedente il task 1y era confuso dal tipo di transizione: T0→T1 aveva pochissimi positivi, mentre T1→T2 ne aveva molti di più.

La nuova domanda è quindi più semplice e clinicamente interpretabile:

> Due scansioni realmente osservate, e in particolare il loro cambiamento, migliorano la prognosi rispetto alla sola ultima scansione osservata?

Questa fase **non predice un embedding futuro**. Isola prima il valore della storia longitudinale reale. COLIPRI rimane completamente congelato.

## Dataset utilizzato

La sorgente resta `pilot_1000_balanced`, coorte NLST deliberatamente arricchita con circa 500 partecipanti ever-cancer e 500 never-cancer. Non rappresenta la prevalenza naturale NLST e non permette di interpretare direttamente le probabilità come rischio assoluto nella popolazione.

Partiamo dalle stesse 1.984 coppie e dagli stessi split per paziente degli esperimenti precedenti. Intersechiamo le coppie T0→T1 e T1→T2 per lo stesso paziente, richiedendo la corrispondenza esatta dell'indice della scansione T1. Il nuovo dataset contiene:

| Split | Pazienti | Campioni |
|---|---:|---:|
| Train | 591 | 591 |
| Validation | 198 | 198 |
| Test | 197 | 197 |
| Totale | **986** | **986** |

Ogni paziente contribuisce una sola riga. Non esiste quindi dipendenza tra due coppie dello stesso paziente nella nuova tabella; lo split resta comunque quello storico a livello paziente.

Input disponibili al modello:

- embedding COLIPRI T0 congelato, 864-d;
- embedding COLIPRI T1 congelato, 864-d;
- intervallo osservato T0–T1 in giorni.

Il modello non riceve T2, l'intervallo futuro T1–T2, il round come variabile esplicita, dati clinici, etichette diagnostiche o informazioni post-T1. T2 è usato soltanto per definire il landmark degli endpoint e come oracle separato.

L'intervallo T0–T1 è simile nei tre split: media circa 372 giorni; range complessivo 238–620 giorni. L'uso di questo intervallo è realistico perché entrambe le scansioni sono già osservate. L'intervallo effettivo T1–T2 non viene fornito: sarebbe informazione futura al momento di T1. Questa scelta rende più difficile prevedere un endpoint ancorato a T2 ed è una limitazione esplicita del disegno.

### Endpoint

Gli endpoint restano esattamente quelli già costruiti: prima diagnosi di cancro entro 365, 730 o 1.095 giorni **dopo T2**. Le etichette mancanti per follow-up insufficiente o ineleggibilità vengono escluse dalla loss e dalle metriche, mai convertite in negativi.

| Endpoint | Train positivi/validi | Validation | Test |
|---|---:|---:|---:|
| Cancro 1y post-T2 | 102/588 (17,35%) | 31/196 (15,82%) | 29/196 (14,80%) |
| Cancro 2y post-T2 | 153/586 (26,11%) | 49/194 (25,26%) | 45/196 (22,96%) |
| Cancro 3y post-T2 | 210/579 (36,27%) | 65/193 (33,68%) | 65/196 (33,16%) |

Questa coorte evita il confronto pooled tra round che produceva da solo AUROC 0,738 a 1 anno. Rimangono però l'arricchimento case-control, il possibile bias di selezione legato alla disponibilità di tutte e tre le scansioni e il fatto che il test sia già stato osservato negli esperimenti precedenti.

## Preprocessing senza leakage

Si definisce il cambiamento osservato:

```text
delta_01 = h_T1 - h_T0
```

Tre standardizzazioni separate vengono stimate esclusivamente sui 591 pazienti di training:

- `h_T1` per lo stato corrente;
- `delta_01` per la variazione osservata;
- `delta_t_01` per l'intervallo temporale.

L'oracle T2 usa un proprio scaler train-only. Validation e test vengono soltanto trasformati. Non si applica PCA, SIGReg o fine-tuning di COLIPRI.

## Architettura primaria: gated observed progression

```text
h_T1 standardizzato [864] ─> Linear(864,256) ─> LayerNorm ─> GELU ─> Dropout ─> s [256]

delta_01 standardizzato [864] -> Linear(864,256) -> LayerNorm -> GELU -> Dropout -> d [256]

delta_t_01 [1] -> Linear(1,32) -> GELU -> Linear(32,32) -> e_t [32]

[s, d, e_t] -> Linear(544,256) -> sigmoid -> gate g [256]

fused = s + g * d
fused -> LayerNorm -> Linear(256,3) -> logits 1y/2y/3y
```

Il gate è feature-wise: può ignorare componenti rumorose della variazione e conservarne altre. Il modello è intenzionalmente piccolo, con dropout 0,1. Non produce una traiettoria latente e non usa T2 durante il training degli input.

## Controlli e ablation predefinite

| Modello | Input | Scopo |
|---|---|---|
| `current_linear` | T1 | baseline lineare storica sulla stessa coorte |
| `current_time_linear` | T1, intervallo T0–T1 | controllo lineare per il tempo |
| `current_time_mlp` | T1, embedding temporale | baseline non lineare senza storia |
| `history_linear` | T1, delta osservato, intervallo | verifica se la progressione è già linearmente utile |
| `history_mlp` | concatenazione T1/delta/tempo | controllo non lineare senza gate |
| `history_gated` | T1 e delta in branche separate, gate e tempo | candidato primario |
| `oracle_t2_linear` | T2 reale | limite superiore osservato, non modello utilizzabile a T1 |

Il confronto principale predefinito è `history_gated` contro `current_time_mlp`. `history_linear` contro `current_time_linear` verifica lo stesso principio con capacità minima. L'oracle T2 non partecipa alla selezione.

Non viene scelto il migliore guardando il test. Tutti i risultati vengono riportati; l'architettura gated è il candidato scientifico primario stabilito in anticipo.

## Addestramento

- PyTorch Lightning, una GPU, W&B online.
- Cinque seed indipendenti per ogni modello.
- AdamW, learning rate 0,001, weight decay 0,0001.
- Batch 128, massimo 150 epoche, early stopping 20.
- Checkpoint sul minimo della BCE prognostica macro di validation.
- BCE multi-label indipendente per 1/2/3 anni.
- Ogni endpoint pesa ugualmente nella media.
- `pos_weight = negativi/positivi` calcolato separatamente sul training.
- Le label mancanti sono mascherate per endpoint.

Gli endpoint sono cumulativi ma le teste non impongono monotonicità. Con BCE pesata, gli output non sono probabilità calibrate; calibrazione e monotonicità richiederebbero un esperimento separato.

## Valutazione

Metriche per seed sullo stesso validation/test set:

- AUROC;
- AUPRC;
- MCC;
- F1;
- sensibilità e specificità alla soglia 0,5, per coerenza con il protocollo storico;
- BCE prognostica.

Le conclusioni principali useranno AUROC e AUPRC. MCC/F1 dipendono dalla soglia e non saranno usati da soli per selezionare una rappresentazione.

Analisi primaria: distribuzione dei risultati sui cinque seed, senza mediare embedding. Analisi secondaria: ensemble delle probabilità dei cinque seed e bootstrap appaiato per paziente, 2.000 ricampionamenti, per le differenze:

- `history_gated - current_time_mlp`;
- `history_linear - current_time_linear`;
- `history_gated - current_linear`.

Poiché esiste una riga per paziente, il bootstrap campiona direttamente i pazienti. Gli intervalli descrivono il campione test fisso, non una validazione esterna né tutta l'incertezza dovuta alla selezione del modello.

## Criterio interpretativo

L'esperimento supporta il valore della progressione osservata soltanto se:

1. il modello con storia migliora il controllo senza storia sulla validation;
2. il guadagno mantiene direzione coerente nel test e in AUROC/AUPRC;
3. il miglioramento non dipende da un solo seed;
4. il bootstrap appaiato è compatibile con un vantaggio clinicamente non trascurabile;
5. il guadagno è presente soprattutto nel confronto matched con lo stesso accesso all'intervallo temporale.

Un miglioramento rispetto a `current_linear` ma non a `current_time_mlp` indicherebbe maggiore capacità del classificatore, non necessariamente valore della storia. Se `history_linear` e `history_gated` non migliorano i rispettivi controlli, non avremo evidenza che il delta COLIPRI globale contenga progressione prognostica utilizzabile.

## Artefatti previsti

Directory: `outputs/colipri_observed_progression_prognosis/`.

- manifest dei 986 pazienti e distribuzione delle classi;
- scaler train-only;
- configurazione e hash di provenienza;
- checkpoint, training history e URL W&B per seed;
- logits/probabilità con patient IDs;
- metriche per seed e tabelle aggregate;
- bootstrap appaiato;
- `status.json`, `run.log`, `COMPLETE.json`;
- `RESULTS.md` generato al completamento.

## Implementazione e stato operativo

File implementati:

- `src/lung_jepa/models/observed_progression.py`: sette architetture con una loss e un protocollo Lightning comuni;
- `scripts/run_colipri_observed_progression.py`: costruzione dei triplet, controlli di allineamento, scaler, training, esportazione, metriche, bootstrap e report finale;
- `configs/model/colipri_observed_progression.yaml`: configurazione completa e seed;
- `scripts/launch_colipri_observed_progression.sh`: lancio persistente in background mediante UUID GPU, PID e blocco contro doppi lanci;
- `tests/test_observed_progression.py`: controlli su shape, gradienti e uso effettivo del delta.

Tre test CPU superati: tutte le architetture producono tre logits e gradienti finiti; i quattro controlli current-only ignorano esattamente il delta; i tre modelli history cambiano output quando cambia il delta. Compilazione Python, sintassi Bash e `git diff --check` superati.

Durante gli smoke test sono stati intercettati e corretti due problemi prima del run completo:

1. il bootstrap della configurazione ridotta assumeva la presenza di baseline escluse dallo smoke; ora esegue soltanto confronti disponibili;
2. gli ID paziente erano inizialmente salvati come array NumPy `object`; ora sono Unicode nativo e gli NPZ si leggono con `allow_pickle=False`.

Le directory dei due smoke intermedi sono conservate con suffisso `failed_20260922_1213` e `object_ids_20260922_1214` per audit, non sono usate nell'analisi. Lo smoke finale ha completato tre training GPU e il bootstrap. Sono stati verificati 986 ID in ordine esatto, logits `[986,3]` finiti e rilettura sicura di tutte le esportazioni. W&B smoke summary: [run pzjncv15](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/pzjncv15).

Prima del lancio, `nvidia-smi` mostrava GPU 3 A100 80GB libera; GPU 0 era occupata da un processo non modificato. La batteria completa è stata avviata in background il 22 settembre 2026:

- PID: `3232852`;
- GPU fisica: 3, NVIDIA A100 80GB;
- UUID: `GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01`;
- training previsti: 7 modelli × 5 seed = **35**;
- W&B project: `nlst-colipri-temporal-adapter`;
- W&B group: `observed-progression-prognosis-v1`;
- log: `outputs/colipri_observed_progression_prognosis/run.log`.

La verifica dopo il lancio mostra il PID corretto sulla GPU 3 con circa 594 MiB allocati; snapshot iniziale: `current_linear`, seed 2, con i seed 0–1 già sincronizzati. Lo stato aggiornato è in `status.json`. Al completamento verranno creati `COMPLETE.json` e `RESULTS.md`; i risultati non vengono anticipati in questo documento di protocollo.
