# LUNG1 E4: hiding GTV tokens inside frozen encoders

**Run date:** 17 October 2026  
**Models:** COLIPRI, TANGERINE, Rad-JEPA-3D, CT-CLIP, DINOv3 (all frozen)  
**Cohort and split:** 416 LUNG1 patients and, for the control comparison, the 403 with a valid control. Same five locked outer folds, inner penalty selection, 16-component PCA and ridge Cox as the [E4 regional probes](lung1_e4_regional_frozen_cox_report.md).

## Question and design

Does a frozen encoder's prognostic signal depend on information that **flows from tumor tokens into the rest of the representation**? The [E4 regional](lung1_e4_regional_frozen_cox_report.md) Non-GTV pool cannot answer this. It drops tumor positions *after* encoding, when the other tokens have already attended to the tumor. Here the CT input is unchanged and the tumor tokens are hidden **inside** the encoder.

1. **Tokens to hide.** These are the native tokens, before any 8³ pooling, that contain any GTV voxel center. GTV voxels come from the cached segmentation resampled onto the original CT grid. The control is the same GTV shape translated by the locked same-axial-level control shift of the [volume experiment](lung1_e4_regional_frozen_cox_report.md#follow-up-gtv-filling-before-the-frozen-encoder). It is mapped the same way and matched to **exactly the same number of tokens**: control-touching tokens are taken by decreasing footprint, then the nearest non-GTV tokens, and never a GTV token.
2. **Hiding mechanism**, per architecture:

   | Encoder | Native grid | Mechanism |
   | --- | --- | --- |
   | TANGERINE (ViT-L, absolute positions) | 16³ | Keys masked in every self-attention layer |
   | COLIPRI (Primus/EVA, 3-D RoPE) | 24³ | Keys masked in every EVA block; the attention pool also ignores hidden tokens |
   | DINOv3 (2-D ViT, RoPE, per slice) | slices × 14 × 14 | Keys masked in every layer of each slice |
   | CT-CLIP (CTViT, spatial + temporal attention) | 24³ | Keys masked in spatial and temporal attention; hidden tokens zeroed at the input of every PEG depthwise 3-D convolution, the only other path that mixes tokens |
   | Rad-JEPA-3D (Mamba + GQA hybrid) | 4 × 16 × 16 | Mamba is a sequential scan with no attention to mask, so hidden tokens are removed from the sequence with the encoder's own JEPA-context `indices` argument. Absolute position embeddings stay correct. Its GQA RoPE uses the sequence index, which is the same behaviour as its pretraining context encoder. |

   Hidden tokens' own outputs are excluded from every readout. For a transformer, masking keys and excluding outputs is the same as dropping the tokens while keeping their positions. The DINOv3 checkpoint's `mask_token` is all zeros (untrained), so mask-token replacement was not used.
3. **Readouts.**
   - **Retained-token mean (primary).** Mean of final native tokens at positions that are visible in every condition: outside GTV tokens (416 patients) or outside GTV and control tokens (403). The token definitions are those of the E2 cache. Original and masked conditions therefore pool **identical positions**; the only difference is whether those tokens could see the tumor.
   - **Official global readout (secondary).** Each encoder's E1 global vector computed over its visible tokens: COLIPRI attention pool, TANGERINE `fc_norm`(CLS ⊕ patch mean), DINOv3 slice-mean CLS, CT-CLIP per-position temporal mean then projection, Rad-JEPA token mean.
4. **Probe.** Every readout × condition is refitted separately. Paired intervals use 300 held-out patient resamples within each fixed fold (seed 20261013), shared across conditions and models.

**Verification.** On a test patient, the unmasked reimplementation reproduces each official forward exactly (relative difference ≤ 5 × 10⁻⁶ for tokens and ≤ 6 × 10⁻⁷ for global vectors). This was rechecked on the first three patients of every model run. The leakage test replaced every input voxel of the hidden patches with uniform noise. With hiding on, retained tokens and global vectors changed by **exactly 0** for all five encoders. With hiding off, the same retained tokens changed by 2% (TANGERINE) to 39% (COLIPRI). On the re-encoded original condition the global vector reproduces the E3 PCA16 score (COLIPRI 0.617, TANGERINE 0.599, Rad-JEPA-3D 0.539, DINOv3 0.575; CT-CLIP 0.528 vs 0.530, the same small drift seen in the volume re-encoding). The all-token mean reproduces the E4 "All" pool.

**Hidden amount.** Median hidden tokens (IQR) and median share of all tokens: COLIPRI 38 (15–74), 0.27%; TANGERINE 15 (8–25), 0.37%; Rad-JEPA-3D 8 (4–13), 0.78%; CT-CLIP 42 (18–86), 0.30%; DINOv3 58 (24–107), 0.24%. The GTV fills a median 24%, 12%, 6%, 26% and 18% of the hidden token volume, so hiding also removes surrounding tissue, most of all for Rad-JEPA-3D's 95 mm deep tokens. The matched control captures a median ≥ 99.9% of its own footprint.

## Results

Mean held-out Uno C through 36 months, PCA16.

**All 416 patients**

| Encoder | Retained mean, original | Retained mean, GTV hidden | Original − hidden (95%) | Global, original | Global, GTV hidden | Original − hidden (95%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | 0.590 | 0.595 | −0.005 [−0.022, +0.014] | 0.617 | 0.606 | +0.011 [−0.007, +0.028] |
| TANGERINE | 0.576 | 0.576 | +0.000 [−0.003, +0.004] | 0.599 | 0.593 | **+0.006 [+0.001, +0.011]** |
| Rad-JEPA-3D | 0.541 | 0.548 | −0.007 [−0.015, +0.000] | 0.539 | 0.548 | **−0.009 [−0.017, −0.001]** |
| CT-CLIP | 0.524 | 0.520 | +0.004 [−0.001, +0.009] | 0.528 | 0.527 | +0.001 [−0.005, +0.008] |
| DINOv3 | 0.546 | 0.545 | +0.001 [−0.004, +0.006] | 0.575 | 0.579 | −0.004 [−0.016, +0.007] |

**Same 403 patients, with the count-matched control** (Uno C; last two columns paired 95% intervals)

| Encoder | Readout | Original | GTV hidden | Control hidden | Original − GTV | Control − GTV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COLIPRI | retained | 0.596 | 0.599 | 0.597 | −0.004 [−0.022, +0.016] | −0.002 [−0.020, +0.019] |
| | global | 0.623 | 0.610 | 0.621 | +0.013 [−0.004, +0.030] | +0.010 [−0.006, +0.030] |
| TANGERINE | retained | 0.571 | 0.572 | 0.571 | −0.001 [−0.005, +0.002] | −0.001 [−0.005, +0.002] |
| | global | 0.594 | 0.588 | 0.592 | **+0.006 [+0.001, +0.011]** | +0.004 [−0.002, +0.008] |
| Rad-JEPA-3D | retained | 0.548 | 0.551 | 0.547 | −0.002 [−0.011, +0.005] | −0.003 [−0.008, +0.001] |
| | global | 0.544 | 0.550 | 0.544 | −0.005 [−0.013, +0.001] | **−0.006 [−0.011, −0.000]** |
| CT-CLIP | retained | 0.524 | 0.521 | 0.521 | +0.002 [−0.002, +0.006] | −0.000 [−0.003, +0.004] |
| | global | 0.517 | 0.516 | 0.517 | +0.001 [−0.006, +0.009] | +0.001 [−0.007, +0.010] |
| DINOv3 | retained | 0.546 | 0.543 | 0.543 | +0.003 [−0.002, +0.008] | +0.000 [−0.006, +0.006] |
| | global | 0.562 | 0.560 | 0.566 | +0.003 [−0.006, +0.011] | +0.007 [−0.004, +0.015] |

**How much the representation moved** (median relative L2 change from the original, 403 patients; retained mean / global):

| Encoder | GTV hidden | Control hidden |
| --- | ---: | ---: |
| COLIPRI | **0.447 / 0.508** | 0.014 / 0.018 |
| TANGERINE | 0.001 / 0.002 | 0.001 / 0.002 |
| Rad-JEPA-3D | 0.026 / 0.027 | 0.026 / 0.027 |
| CT-CLIP | 0.006 / 0.023 | 0.005 / 0.024 |
| DINOv3 | 0.007 / 0.017 | 0.007 / 0.016 |

**Sensitivity, PCA8.** Most contrasts stay near zero. Two change:
- COLIPRI's retained-token score *rises* when the GTV is hidden (0.558 → 0.584 on 403 patients; control − GTV −0.029 [−0.058, −0.000]), the opposite sign of its PCA16 global drop.
- TANGERINE's global contrasts strengthen (original − GTV +0.008 [+0.002, +0.014]; control − GTV +0.005 [+0.001, +0.010]).

Full tables are in the `_pca8` output directories.

## Interpretation

- **Hiding the tumor barely changes survival discrimination.** With PCA16, every contrast is below 0.015 Uno C in absolute value; with PCA8, only COLIPRI's retained-token contrast is larger. Among the primary retained-token contrasts, not one interval excludes zero for any encoder. The small effects that reach nominal significance go in opposite directions (TANGERINE global slightly worse, Rad-JEPA-3D slightly better) and come without multiplicity adjustment.
- **COLIPRI is the exception in representation, not in prognosis.** Hiding about 0.3% of its tokens at the tumor moves its retained-token mean and global vector by about 45–51%. Hiding the same number of tokens elsewhere moves them by under 2%. COLIPRI's tokens therefore route a large share of their content through tumor tokens, unlike the other four encoders. Even so, its held-out discrimination changes by at most about 0.01 at PCA16 and has the opposite sign at PCA8. Its linear Cox probe can recover nearly the same ranking from the tumor-blind representation.
- **Consistency with the earlier E4 results.** COLIPRI's PCA16 global drop under hiding (+0.011, n.s.) is in the same direction as, and smaller than, its drop under shell-median filling (+0.018) and black filling (+0.023), and is close to inpainting (+0.003). All of these depend on the intervention or the probe capacity. Rad-JEPA-3D's weak tumor-position pool in E4 regional (0.472) does not mean that it needs tumor context: hiding its tumor tokens does not hurt it.
- **Limits.** Masking removes whole tokens, so it also hides peritumoral tissue. For Rad-JEPA-3D's coarse tokens this tissue is the majority. The encoders never saw this kind of contiguous hiding at pretraining, except Rad-JEPA-3D, whose JEPA context path drops tokens. DINOv3 is changed only on tumor-containing slices. As with all E4 analyses, this is internal cross-validation conditional on fixed folds and fitted models, not external validation.

## Reproduction and artifacts

```bash
cd scripts
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
  ../.venv/bin/python audit_lung1_e4_native_token_mapping.py --workers 8
cd .. && CUDA_VISIBLE_DEVICES=<A100 UUID> bash scripts/launch_lung1_e4_token_masking.sh   # all five encoders, ~2 h
cd scripts
for flag in "" --control; do for k in 16 8; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python run_lung1_e4_token_masking_cox.py $flag --components $k
done; done
```

- Native token masks (GTV, count-matched control, footprints): `outputs/lung1_e4_token_masking/geometry/masks/<model>/<patient>.npz`, summary `geometry/patient_model_geometry.csv`.
- Per-patient readout vectors and change/verification records: `outputs/lung1_e4_token_masking/embeddings/<model>/{vectors,patients}`, `patient_summary.csv`.
- Cox fits, summaries, `paired_mask_differences.csv`, `embedding_change_summary.csv` and `token_masking_uno_c.png`: `outputs/lung1_e4_token_masking/cox_{full416,control403}{,_pca8}`.
