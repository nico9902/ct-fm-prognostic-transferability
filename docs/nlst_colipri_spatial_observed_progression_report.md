# Progressione spaziale COLIPRI osservata — protocollo

Data: 22 settembre 2026. Protocollo definito prima del run completo; i risultati verranno generati separatamente in `outputs/colipri_spatial_observed_progression/RESULTS.md`.

## Domanda

Il delta globale mean-pooled T1−T0 non ha migliorato la prognosi post-T2. Questo esperimento verifica se la media abbia cancellato variazioni focali: usa la progressione **realmente osservata** nei 64 token spaziali, senza predire il futuro e senza Transformer.

## Coorte ed endpoint

Stessi 986 pazienti con T0, T1 e T2 dell'esperimento globale, una riga per paziente e split invariati: 591 train, 198 validation, 197 test. La coorte `pilot_1000_balanced` è arricchita ever-cancer/never-cancer e non riflette la prevalenza NLST naturale.

Input: token COLIPRI congelati T0 e T1 `[64,864]`, delta osservato T1−T0 e intervallo T0–T1. Endpoint: prima diagnosi entro 1/2/3 anni dopo T2. Positivi train: 102/153/210; test: 29/45/65. T2 entra solo negli oracle.

Il cache contiene `[2984,64,864]`. La media dei token ricostruisce l'embedding globale con MSE media `1,32×10⁻¹⁶`; il runner ripete il controllo e interrompe il training in caso di mismatch.

## Modello primario

Non c'è self-attention tra token. Un MLP condiviso proietta ogni token T1 da 864 a 128 dimensioni; un secondo MLP condiviso proietta il delta. Un gate feature-wise decide quanto delta usare per ciascun token. Un attention pooling scalare assegna un peso ai 64 token e produce un vettore globale; una testa lineare restituisce i tre logits.

```text
T1 token [864] -> Linear 128 -> LN -> GELU ---- state_i --+
delta_i [864]  -> Linear 128 -> LN -> GELU ---- change_i -+-> gate_i
dt01 -> MLP 32 -------------------------------------------+

fused_i = state_i + gate_i * change_i
attention_i = softmax(score(fused_i))
pooled = sum_i attention_i * fused_i
[pooled, time] -> Linear -> logits 1y/2y/3y
```

Dropout 0,1; circa 250 mila parametri, da registrare esattamente in `provenance.json`. COLIPRI resta congelato.

## Controlli

- `mean_linear`: media token T1 + tempo, baseline globale.
- `token_attention_t1`: T1 con attention pooling, senza delta.
- `token_gated_control`: architettura gated completa, ma delta paziente azzerato.
- `token_gated_history`: stessa identica architettura e numero di parametri, con delta reale.
- `oracle_mean_linear`: media dei token T2 reali.
- `oracle_token_attention`: token T2 reali con attention pooling.

Confronto primario predefinito: `token_gated_history - token_gated_control`. Questo isola il valore del delta senza confonderlo con la capacità. Confronti secondari: attention T1 vs mean T1, history vs attention T1, oracle token vs oracle mean.

## Preprocessing e training

StandardScaler feature-wise separati per token T1, delta e T2 oracle, stimati solo sui pazienti train aggregando i 64 token. Tempo standardizzato sul train. Nessuna PCA, SIGReg, label diagnostica o feature futura.

PyTorch Lightning, AdamW lr 0,001, weight decay 0,0001, batch 32, massimo 150 epoche, patience 20, BCE multi-endpoint mascherata e pesata per classe. Cinque seed per sei modelli: 30 training. Checkpoint sul minimo della BCE prognostica macro di validation. W&B project `nlst-colipri-temporal-adapter`, gruppo `spatial-observed-progression-v1`.

## Valutazione e artefatti

AUROC, AUPRC, MCC, F1, sensibilità, specificità e BCE per seed. Analisi secondaria con ensemble delle probabilità e 2.000 bootstrap appaiati per paziente. Le probabilità BCE pesate non sono calibrate; il test è già stato osservato in precedenti esperimenti e non costituisce validazione esterna.

Vengono salvati checkpoint, storie, scaler, manifest, hash, logits, probabilità, attenzione `[986,64]` e gate medio `[986,64]`, metriche, bootstrap, log, stato e report finale. Le mappe sono segnali di importanza del modello, non segmentazioni o spiegazioni causali.

File implementati:

- `src/lung_jepa/models/spatial_progression.py`
- `scripts/run_colipri_spatial_observed_progression.py`
- `configs/model/colipri_spatial_observed_progression.yaml`
- `scripts/launch_colipri_spatial_observed_progression.sh`
- `tests/test_spatial_progression.py`

Tre controlli CPU superati: shape/gradienti/probabilità di attenzione; controllo gated indipendente dal delta; identico numero di parametri fra controllo e history e sensibilità della history al delta. Compilazione Python, sintassi Bash e diff check superati.

## Stato operativo

Smoke test GPU completato con tre modelli e W&B: [summary run 8c4w55c3](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/8c4w55c3). Verificati esattamente 986 patient ID, logits `[986,3]` finiti, attention `[986,64]` con somma unitaria, gate `[986,64]` limitati a `[0,1]` e lettura NPZ con `allow_pickle=False`. Il controllo e il modello history hanno entrambi **269.380 parametri**. Il nuovo audit indipendente del pooling riporta MSE media `3,75×10⁻¹⁶`, massimo `8,14×10⁻¹⁶` e cosine minimo `0,9999997`.

La batteria completa è stata avviata in background il 22 settembre 2026:

- PID `3250797`;
- GPU fisica 3, NVIDIA A100 80GB;
- UUID `GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01`;
- 6 modelli × 5 seed = 30 training;
- W&B group `spatial-observed-progression-v1`;
- log `outputs/colipri_spatial_observed_progression/run.log`.

La verifica `nvidia-smi` mostra il PID corretto sulla GPU 3, circa 618 MiB allocati. Snapshot iniziale: `mean_linear`, seed 0. Lo stato aggiornato è in `status.json`; `RESULTS.md` e `COMPLETE.json` verranno prodotti automaticamente al termine.
