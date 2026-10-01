# Audit di allineamento longitudinale dei token COLIPRI

Data: 22 settembre 2026. Audit completato su tutte le 1.984 coppie longitudinali, senza utilizzare outcome o label prognostiche.

## Obiettivo

Verificare se il fallimento prognostico del delta spaziale fosse causato dalla sottrazione di token non corrispondenti anatomicamente. Il cache congelato contiene 2.984 scansioni, ciascuna rappresentata da una griglia nativa COLIPRI `[4,4,4,864]`, appiattita in `[64,864]`.

## Protocollo

Sono state analizzate separatamente T0→T1 e T1→T2, per train/validation/test e in due spazi:

- `raw`: cosine sui token COLIPRI nativi;
- `standardized`: standardizzazione feature-wise stimata soltanto sulle scansioni dei pazienti train, poi cosine.

Per ogni matrice di similarità `64×64` sono stati calcolati:

- cosine dello stesso indice spaziale;
- nearest-neighbor locale, entro raggio Chebyshev 1;
- nearest-neighbor globale;
- percentuale in cui lo stesso token è top-1;
- mutual nearest-neighbor e margine top-1/top-2;
- matching univoco Hungarian;
- traslazione globale della griglia tra le 27 possibilità `{-1,0,1}³`.

La traslazione è selezionata sui token a parità pari e valutata sui token a parità dispari, evitando di misurare il guadagno sugli stessi token usati per sceglierla. Cinque controlli casuali associano ogni sorgente a un altro paziente dello stesso split e della stessa transizione, usando derangement senza auto-associazioni.

## Risultati principali nel test

| Spazio / transizione | Cosine stessa posizione | Stesso token top-1 | Hungarian posizioni fisse | Traslazione zero | Guadagno locale | Guadagno traslazione held-out |
|---|---:|---:|---:|---:|---:|---:|
| Raw T0→T1 | 0,958 | 80,5% | 97,7% | 99,0% | 0,0023 | 0,0001 |
| Raw T1→T2 | 0,956 | 79,4% | 99,1% | 99,5% | 0,0026 | 0,0002 |
| Standardized T0→T1 | 0,823 | 84,4% | 98,9% | 99,0% | 0,0057 | 0,0004 |
| Standardized T1→T2 | 0,822 | 83,8% | 99,3% | 99,5% | 0,0061 | 0,0007 |

Il confronto con pazienti casuali conferma che la similarità non deriva soltanto da caratteristiche comuni del polmone:

| Spazio / transizione | Coppia vera, stessa posizione | Paziente casuale | Differenza |
|---|---:|---:|---:|
| Raw T0→T1 | 0,958 | 0,857 | +0,101 |
| Raw T1→T2 | 0,956 | 0,847 | +0,109 |
| Standardized T0→T1 | 0,823 | 0,433 | +0,390 |
| Standardized T1→T2 | 0,822 | 0,410 | +0,412 |

Il matching globale aumenta il cosine standardizzato soltanto di circa 0,008. Il matching locale recupera circa 0,006. La traslazione selezionata su metà griglia produce un guadagno held-out inferiore a 0,001.

## Robustezza e outlier

Nello spazio standardizzato, la mediana della percentuale top-1 sulla stessa posizione è 89,1% per entrambe le transizioni; il 5° percentile è 54,7% per T0→T1 e 51,6% per T1→T2. Il matching Hungarian mantiene il 100% delle posizioni almeno fino al 5° percentile; una piccola coda, circa 1%, contiene scansioni molto meno stabili.

La similarità diagonale diminuisce debolmente con l'intervallo temporale: Spearman circa −0,10 per T0→T1 e −0,12 per T1→T2; per la percentuale top-1 circa −0,11 e −0,14. Non è un effetto abbastanza grande da spiegare il risultato prognostico complessivo.

## Conclusione

L'ipotesi che il delta token-level fallisca principalmente per disallineamento della griglia **non è supportata**. Alla risoluzione 4×4×4, i token COLIPRI mantengono una corrispondenza longitudinale forte e prevalentemente identitaria. Una registrazione rigida della griglia o un nearest-neighbor matching modificherebbero pochissimo la maggioranza dei campioni.

Questo audit non dimostra corrispondenza anatomica voxel-perfect: token semanticamente simili possono coincidere anche senza registrazione deformabile. Tuttavia, la combinazione di top-1, Hungarian, controllo casuale e traslazione held-out rende improbabile che un semplice errore di indice sia il collo di bottiglia dominante.

Il fallimento del delta spaziale va quindi attribuito più plausibilmente a uno o più fattori differenti:

- cambiamenti precedenti poco predittivi di nuove alterazioni tra T1 e T2;
- delta dominato da variazioni biologiche o tecniche non oncologiche, pur nella posizione corretta;
- griglia 4×4×4 troppo grossolana per noduli focali;
- numerosità insufficiente per imparare quali piccole componenti del delta siano clinicamente rilevanti;
- forte informazione diagnosis-adjacent presente soltanto nel T2 reale.

Non è giustificato lanciare subito una batteria con token riallineati: il beneficio massimo osservato dal matching è troppo piccolo. È più informativo studiare risoluzione locale/lesion-level ed endpoint con lag dalla diagnosi.

## Artefatti

Directory `outputs/colipri_token_alignment_audit/`:

- `alignment_per_pair.csv`: metriche per coppia e spazio;
- `alignment_summary.csv`: aggregati per split e transizione;
- `translation_counts.csv`: traslazioni selezionate;
- `feature_scaler.npz`: statistiche train-only;
- `cosine_alignment.png`, `top1_displacement.png`: figure;
- `RESULTS.md`, `COMPLETE.json`, configurazione, fingerprint, log e provenienza.

Implementazione: `scripts/run_colipri_token_alignment_audit.py`. Test sintetici superati per identità, traslazione nota e derangement entro gruppo. W&B: [token alignment audit](https://wandb.ai/domenico-paolo1999/nlst-colipri-temporal-adapter/runs/libqs93w).
