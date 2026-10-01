# Guida alla presentazione COLIPRI — 23 settembre 2026

## Obiettivo della riunione

Portare il gruppo a una decisione: prima di aumentare la complessità del world model, conviene investire nella definizione temporale dell’endpoint e in una rappresentazione locale/lesion-aware. La tesi da sostenere è:

> COLIPRI rappresenta bene lo stato clinico e il futuro reale contiene un forte segnale prognostico; i dati disponibili non mostrano però che quel segnale sia ricostruibile dal passato tramite una semplice dinamica globale.

## Durata consigliata

- Versione completa: 28–32 minuti più domande.
- Versione breve: 15 minuti, usando le slide 1–4, 7–9, 12–19, 23–30.
- Le slide 30–32 sono di appendice e possono restare nascoste durante l’esposizione.

## Narrazione in cinque blocchi

### 1. Dataset e validità sperimentale — 5 minuti

Slide 1–8. Spiegare la coorte arricchita, lo split per paziente e le label landmark-safe. Fermarsi sullo squilibrio fra transizioni: nel test 1y ci sono 2 positivi T0→T1 e 29 T1→T2; il solo round raggiunge AUROC 0,738. Questo è il principale caveat per i risultati pooled a breve orizzonte.

### 2. Scelta del foundation model — 5 minuti

Slide 9–13. Rad-JEPA è quasi casuale sui target focali; TANGERINE migliora soprattutto i fenotipi diffusi; COLIPRI è il primo spazio che rende accessibili tutti e quattro i target con un probe lineare. Mostrare che la rappresentazione globale è un mean pooling della griglia densa, non un token clinico addestrato ad hoc.

### 3. Upper bound e world model — 8 minuti

Slide 14–19. Il futuro reale porta AUROC da 0,715 a 0,870 a 1 anno e da 0,654 a 0,742 a 2 anni. Il residual MLP standardizzato batte finalmente l’identità in MSE, ma guadagna pochissimo in prognosi. Il Transformer spaziale nativo migliora la ricostruzione e i target focali, ma non supera chiaramente il pooled MLP.

### 4. Audit delle ipotesi — 7 minuti

Slide 20–24. Separare bene quattro conclusioni:

- i risultati near-chance del primo adapter erano invalidi per un bug di allineamento;
- l’adapter residuale corretto preserva informazione, ma non aggiunge prognosi;
- SIGReg non risolve il problema;
- neppure il delta realmente osservato T0→T1 aiuta a predire dopo T2;
- la griglia 4×4×4 è già longitudinalmente ben allineata.

### 5. Decisione successiva — 5 minuti

Slide 25–29. Il segnale mancante è plausibilmente piccolo, focale, stocastico e diagnosis-adjacent. Proporre un esperimento lesion-aware con controllo current-only vs current+history a capacità identica, endpoint stratificati per lag diagnostico e nessuna previsione completa dell’embedding nella prima fase.

## Frasi chiave

- “La metrica di ricostruzione migliora, ma non è un surrogato della prognosi.”
- “L’oracle gap esiste; il problema è stabilire quale parte sia prevedibile dal passato.”
- “PCA era dannosa, ma rimuoverla non ha risolto la transizione.”
- “La storia osservata non aiuta automaticamente: questo riduce la probabilità che basti un modello più potente.”
- “Non aumenterei la profondità del Transformer prima di dimostrare valore nella storia locale reale.”

## Domande probabili e risposte brevi

### Perché il futuro reale è così forte a 1 anno?

Una parte dei casi è vicina alla diagnosi: nel test 1y, 24 dei 31 positivi sono diagnosticati entro 180 giorni da t+1. La scansione futura può quindi contenere malattia già visibile. Questo è clinicamente reale, ma rende il segnale difficile da prevedere un anno prima.

### Perché una buona MSE non porta buona AUROC?

La MSE premia la ricostruzione delle direzioni con maggiore energia e più comuni. Il segnale oncologico può essere raro, locale e a bassa varianza; un modello può migliorare l’errore medio ignorandolo.

### Serve un Transformer più grande?

Non è la prima scelta supportata dai dati. Il Transformer spaziale nativo ha circa lo stesso risultato prognostico del pooled MLP, e il delta reale osservato non migliora il controllo current-only. Con circa 2.000 coppie, più capacità aumenta il rischio di overfitting.

### L’adapter 128-d ha davvero distrutto la prognosi?

La prima evidenza near-chance era invalidata da un bug feature-label. La rivalutazione corretta mostra comunque che il bottleneck non residuale perde informazione rispetto allo spazio nativo. Un adapter residuale 864-d la preserva quasi interamente, ma non crea un vantaggio prognostico.

### Dobbiamo usare SIGReg?

Non come intervento principale. Nella batteria corretta SIGReg non ha prodotto un miglioramento stabile. Può essere una regolarizzazione secondaria, non una soluzione alla mancanza di segnale prevedibile.

### I token sono disallineati fra scansioni?

Non in modo sufficiente a spiegare il fallimento. Il cosine standardizzato alla stessa posizione è circa 0,823, l’84% dei token mantiene lo stesso nearest neighbor e matching/traslazione aggiungono molto poco.

### Qual è il prossimo esperimento più informativo?

Verificare prima se la storia locale reale contiene segnale: token più fini o regioni candidate, confronto current-only vs current+history con capacità matched, stratificazione per round e per distanza dalla diagnosi. Solo dopo, predire quelle componenti con un world model.

## Chiusura suggerita

“Abbiamo validato lo spazio e l’upper bound, e abbiamo escluso diverse spiegazioni semplici del gap. Il prossimo investimento non dovrebbe essere un modello più grande, ma un target temporalmente più pulito e una rappresentazione più vicina alla lesione.”
