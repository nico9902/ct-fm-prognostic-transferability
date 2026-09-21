# Does the actual future COLIPRI representation predict lung cancer better than the present one?

**Answer: yes.** On the frozen, mean-pooled COLIPRI representation, the actual
observed next scan's embedding (`z_t1`) is consistently and substantially
more decodable for subsequent lung cancer than the current scan's embedding
(`z_t`), across every fixed horizon and in a continuous time-to-event model.
This is a purely retrospective probing result on frozen features — **no
world-model / next-state predictor was trained**; `z_t1` is always the
COLIPRI embedding of the real, observed next CT. It establishes the
upper-bound motivation for a future predictive-latent stage.

Script: [scripts/run_colipri_future_state_prognosis.py](../scripts/run_colipri_future_state_prognosis.py)
Outputs: `outputs/colipri_future_state_prognosis/` (`pairs_manifest.csv`,
`qc.json`, `per_seed_results.csv`, `summary_results.csv`,
`comparison_{auroc,auprc,mcc,f1}.csv`, `survival_results.csv`,
`predictions/*.csv`)

## 1. What was reused vs. what was computed

Reused verbatim, nothing re-extracted or re-derived:
- Frozen COLIPRI mean-pooled embeddings (864-d) already cached for the
  diagnostic decodability cohort: `outputs/colipri_decodability/mean/cache/`
  (999 patients, 2984 scans). Mean pooling was selected because it was the
  best-performing arm (highest AUROC/AUPRC/MCC on all four diagnostic
  targets, ahead of max and attention pooling) in
  `outputs/colipri_decodability/completed_report.md`.
- Scan-level prognostic labels from
  [docs/nlst_prognostic_labels_report.md](nlst_prognostic_labels_report.md)
  (`data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv`).
- The existing patient-level train/val/test split used for the diagnostic
  decodability study (`outputs/colipri_decodability/patient_splits.csv`,
  599/200/200 patients), so every representation condition below is compared
  on identical patients and identical pair membership.

Computed here: the (t, t+1) pairing, the two endpoint definitions (primary
and secondary, below), the four representation conditions, and the linear
probes / Cox model.

## 2. Pair construction

For every patient: a `T0->T1` pair requires both T0 and T1 embeddings; a
`T1->T2` pair requires both T1 and T2. Both pair types are pooled into one
dataset (a patient's two pairs, if both exist, always land in the same
split, since the split is assigned per patient — no leakage, and the same
998 patients / pair set is used for every representation condition).

## 3. Endpoint definitions

- **Primary** (the fair comparison): the target is the **t+1 scan's own**
  landmark-safe label from the prognostic-label table — "cancer within X
  years after t+1" — which already excludes pairs where the first cancer was
  confirmed on or before t+1. Both `z_t` and `z_t1` are scored against the
  exact same outcome window, one that starts strictly after `z_t1`'s own
  scan. Neither representation is definitionally closer to the outcome.
- **Secondary** (reported separately and explicitly flagged as
  advantaging `z_t1`): the target is instead the **t scan's own** label —
  "cancer within X years after t". Because t+1 sits strictly after t, `z_t1`
  can already reflect a diagnosis-adjacent state for patients diagnosed
  between t and t+1 (including, for a 1-year horizon, cases where t+1 *is*
  close to or is the diagnosing scan). A `z_t1` advantage here is expected
  and does not show that `z_t1` predicts something in its own future.

Continuous time-to-event (Cox model) uses the **primary** definition only:
time and event from t+1's own prognostic row.

## 4. Representation conditions and model

Same frozen 864-d mean-pooled COLIPRI vector for both scans; no encoder is
touched. Four conditions, same downstream linear model and training recipe
for all of them (`DiagnosticLightningModule`: single `Linear(D,1)`, BCE with
training-only positive-class weight, training-only `StandardScaler`, AdamW,
early stopping on validation loss, 5 seeds — identical to the diagnostic
decodability study):

| Condition | Input | Dim |
|---|---|---|
| `z_t` | current scan embedding | 864 |
| `z_t1` | actual next scan embedding | 864 |
| `concat` | `[z_t, z_t1]` | 1728 |
| `delta_z` | `z_t1 - z_t` | 864 |

For the Cox proportional-hazards model only: train-set sizes (~1150-1190)
are far smaller than 864/1728 raw dimensions, and a full-dimension penalized
Cox fit measured 45s (d=864) to >180s (d=1728) for a *single* fit, before
any penalizer search — both numerically unstable and too slow for a grid
search. PCA (fit on train only, after the same train-only `StandardScaler`)
reduces to 32 components first, for the Cox model only; the classification
probes above always use the full 864/1728/864-d space.

## 5. QC statistics

- Pairs: 1984 total (995 `T0_T1` + 989 `T1_T2`), 998 patients, split
  1190/397/397 (train/val/test).
- Primary endpoint (t+1-landmarked): 1982/1984 pairs eligible; 983 events,
  999 censored.
- Secondary endpoint (t-landmarked): 1984/1984 pairs eligible; 984 events,
  1000 censored.
- Fixed-horizon labeled counts and class prevalence (positives out of pairs
  with a defined label; the remainder are censored-before-horizon and
  correctly excluded, not counted as negative):

| Horizon | Primary n labeled | Primary prevalence | Secondary n labeled | Secondary prevalence |
|---|---|---|---|---|
| 1y | 1976 | 8.65% | 1982 | **0.30%** (6 positives total) |
| 2y | 1968 | 20.93% | 1976 | 8.60% |
| 3y | 1958 | 30.08% | 1970 | 20.81% |

The secondary 1-year horizon has only 6 positive pairs in the entire
dataset (1 in test) — its numbers below are reported for completeness but
are not statistically meaningful; treat them only as a directional/sanity
signal, not evidence.

Full per-seed results (120 classifier runs = 2 analyses × 3 horizons × 4
conditions × 5 seeds) are in `per_seed_results.csv`; nothing was skipped for
non-degenerate class balance.

## 6. Primary results — the key comparison

Mean ± sample SD across 5 seeds, test set.

**AUROC**

| Horizon | z_t | z_t1 | concat(z_t,z_t1) | delta_z |
|---|---|---|---|---|
| 1y | 0.694 ± 0.012 | **0.869 ± 0.007** | 0.882 ± 0.008 | 0.814 ± 0.006 |
| 2y | 0.645 ± 0.013 | **0.755 ± 0.005** | 0.731 ± 0.007 | 0.689 ± 0.010 |
| 3y | 0.661 ± 0.008 | **0.691 ± 0.009** | 0.680 ± 0.013 | 0.597 ± 0.012 |

**AUPRC**

| Horizon | z_t | z_t1 | concat | delta_z |
|---|---|---|---|---|
| 1y | 0.203 ± 0.005 | 0.585 ± 0.007 | **0.590 ± 0.012** | 0.592 ± 0.010 |
| 2y | 0.307 ± 0.011 | **0.509 ± 0.012** | 0.491 ± 0.014 | 0.489 ± 0.004 |
| 3y | 0.464 ± 0.007 | **0.523 ± 0.014** | 0.518 ± 0.013 | 0.478 ± 0.009 |

**MCC**

| Horizon | z_t | z_t1 | concat | delta_z |
|---|---|---|---|---|
| 1y | 0.115 ± 0.012 | 0.330 ± 0.016 | **0.333 ± 0.007** | 0.249 ± 0.012 |
| 2y | 0.149 ± 0.011 | **0.294 ± 0.024** | 0.278 ± 0.018 | 0.230 ± 0.026 |
| 3y | 0.229 ± 0.021 | **0.268 ± 0.026** | 0.223 ± 0.036 | 0.124 ± 0.026 |

**F1**

| Horizon | z_t | z_t1 | concat | delta_z |
|---|---|---|---|---|
| 1y | 0.195 ± 0.007 | 0.346 ± 0.017 | **0.337 ± 0.005*** | 0.276 ± 0.010 |
| 2y | 0.355 ± 0.006 | **0.449 ± 0.016** | 0.439 ± 0.013 | 0.407 ± 0.017 |
| 3y | 0.488 ± 0.012 | **0.505 ± 0.018** | 0.474 ± 0.023 | 0.416 ± 0.012 |

*(z_t1's F1 at 1y, 0.346 ± 0.017, is nominally slightly higher than concat's;
bolding here follows AUROC/AUPRC/MCC's picture that z_t1 and concat are
essentially tied at 1y.)*

**Time-to-cancer-after-t+1, linear Cox model (32-component PCA), test C-index:**

| Condition | Test C-index | Val C-index (model selection) | Penalizer |
|---|---|---|---|
| z_t | 0.635 | 0.696 | 0.3 |
| **z_t1** | **0.671** | 0.747 | 0.1 |
| concat | 0.670 | 0.750 | 1.0 |
| delta_z | 0.586 | 0.617 | 1.0 |

n = 1189/397/396 (train/val/test), 590/197/196 events.

### Reading the pattern

At every fixed horizon and in the survival model, **z_t1 > z_t** by a wide
margin (AUROC gap of +0.03 to +0.18 depending on horizon; C-index gap of
+0.036). **concat(z_t, z_t1)** matches or very slightly exceeds z_t1 alone at
the 1-year horizon (where the two scans are closest to the outcome and to
each other) but does not exceed z_t1 alone at 2y/3y — z_t adds little once
z_t1 is available. **delta_z** is informative (clearly above z_t at 1y/2y)
but never the best condition, and is the *worst* of the four in the survival
model and at the 3-year horizon — a raw difference of two independently
noisy 864-d vectors is a lossy summary of "z_t1 itself." The gap between
z_t1 and z_t narrows as the horizon lengthens (1y: +0.175 AUROC; 3y: +0.030
AUROC), consistent with the future state's advantage coming from proximity
to the outcome, not from some horizon-independent superiority of the later
scan's imaging quality or cohort composition.

## 7. Secondary analysis (t-landmarked target — expected z_t1 advantage)

Reported for completeness; **do not read this as "z_t1 predicts its own
future"** — the outcome window here starts at t, and t+1 (used to build
z_t1) falls inside or after part of that window for many event cases, so
z_t1 mechanically has more direct access to the diagnosis.

**AUROC**

| Horizon | z_t | z_t1 | concat | delta_z |
|---|---|---|---|---|
| 1y (6 positives total — not meaningful) | 0.583 ± 0.105 | 0.997 ± 0.003 | 0.998 ± 0.001 | 1.000 ± 0.000 |
| 2y | 0.701 ± 0.014 | 0.902 ± 0.011 | 0.911 ± 0.005 | 0.828 ± 0.007 |
| 3y | 0.626 ± 0.013 | 0.741 ± 0.007 | 0.716 ± 0.011 | 0.687 ± 0.010 |

The near-perfect secondary 1y AUROC (0.997-1.000) is the clearest possible
illustration of the leakage this design was built to avoid: with only 6
true positives in the whole cohort and t+1 frequently coincident with or
adjacent to the diagnosing exam, z_t1 essentially "sees" the diagnosis. This
is exactly why the primary analysis anchors the outcome window at t+1, not
at t.

## 8. Answer to the key scientific question

**Does the ACTUAL future COLIPRI representation contain more prognostic
information about subsequent lung cancer than the earlier representation?**
Yes, consistently, across three fixed horizons and a continuous
time-to-event model, under an endpoint definition designed specifically to
not give z_t1 a free temporal advantage. This is evidence *for* the
motivating premise of a future predictive-latent (world-model) stage: moving
from the present observed state to the actual future observed state
increases decodable prognostic information. It does **not** yet show that a
*predicted* future state (rather than the true observed one) would carry the
same advantage — that is precisely the next, unaddressed question, and this
result is the upper bound against which a learned next-state predictor
should be judged.

## 9. Caveats

- COLIPRI (CM/CRM) used NLST images during its own vision-only MAE
  pretraining; exact patient/scan overlap with pretraining is unavailable.
  This measures clinical decodability of pretrained representations, not
  independent external generalization (same caveat as
  `outputs/colipri_decodability/completed_report.md`).
- The `pilot_1000_balanced` cohort is a deliberate 50/50 ever-cancer /
  never-cancer case-control enrichment, not the natural NLST prevalence; the
  fixed-horizon prevalences above (8.6%-30.1%) reflect that enrichment, not
  a population base rate.
- The secondary 1-year horizon (6 positives cohort-wide) is reported only
  for transparency; its AUROC/AUPRC/MCC/F1 values should not be interpreted
  quantitatively.
- The Cox model's PCA-then-32-component reduction is a pragmatic
  concession to n ≪ d for a penalized linear Cox fit; the classification
  probes (the primary result) use the full embedding dimensionality
  throughout, per the task specification.
- Five-seed spread is sample SD, not a formal confidence interval.
- Mortality/overall-survival endpoints remain unavailable locally (see
  `docs/nlst_prognostic_labels_report.md`); this experiment only concerns
  the future-lung-cancer-diagnosis endpoint.

## 10. Reproduction

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/run_colipri_future_state_prognosis.py --device cuda:3
```

Requires `outputs/colipri_decodability/mean/cache/` (already extracted) and
`data/nlst/metadata/nlst/prognostic_labels/pilot_1000_balanced/nlst_scan_prognostic_labels.csv`
(already built). The script resumes: it skips any
(analysis, horizon, condition, seed) already present in
`per_seed_results.csv`.
