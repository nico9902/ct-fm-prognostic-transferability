# E1 sensitivity experiment: GTV preserving thorax crop and full embedding Cox

## Input audit before the encoders

The new input source is `data/processed/lung1/thorax_hu/`: a 1 mm isotropic bounding box around the union of both lungs and the GTV, with a 10 mm margin. The stored float32 CT is still in HU. The lung window used for a preprocessing preview is **not** stored in these volumes. All 416 eligible crops have `gtv_fraction_inside_crop = 1.0` in their metadata. The crop uses an annotation to set its field of view; this matters for interpretation and should be compared with an annotation-free policy in a later experiment.

For LUNG1-001, the stored crop is 317 × 234 × 281 voxels (X × Y × Z), at 1 mm spacing, and its HU range is −1024 to 2992.8. The following image was made from the exact normalized tensors immediately before encoder inference; the selected slices pass near the GTV centroid. The GTV was used to choose the display slices, not added to the input tensor.

![LUNG1-001 normalized encoder inputs](../outputs/lung1_e1_input_examples/thorax_hu/LUNG1-001_comparison.png)

The per-model files and exact tensor statistics are in [`LUNG1-001_input_audit.json`](../outputs/lung1_e1_input_examples/thorax_hu/LUNG1-001_input_audit.json).

| Frozen encoder | Final input tensor `(C,D,H,W)` | Intensity operation after crop | Global vector passed to Cox |
| --- | --- | --- | ---: |
| TANGERINE | `(1,256,256,256)` | B-spline resize; clip HU to [−1200, 800]; `(HU + 1200) / 2000` | 2048 |
| Rad-JEPA-3D | `(1,32,256,256)` | RAS orientation; clip HU to [−1000, 400]; `(HU + 1000) / 1400`; linear resize | 384 |
| COLIPRI | `(1,192,192,192)` | Official processor: RAS, 2 mm resampling, clip HU to [−1000, 1000], map to [−1, 1], center crop or pad | 768 |
| CT-CLIP | `(1,240,480,480)` | LPS; 0.75 × 0.75 × 1.5 mm resampling; clip HU to [−1000, 1000], divide by 1000, center crop or pad | 512 |

The extra leading batch dimension at inference makes each input `(1,C,D,H,W)`. The sample normalized ranges were TANGERINE [0.0753, 1], Rad-JEPA [0, 1], COLIPRI [−1, 1], and CT-CLIP [−1, 1]. Thus clipping and normalization still occur, exactly once in the encoder input path. Source crop HU values above each model's upper window are saturated there, as prescribed by that model's input transform.

## Survival probes

The original E1 probe used standardized embeddings, 32 PCA components and L2 Cox regression. PCA is unsupervised: the directions with the most feature variance need not be the directions with the most survival information. The new primary sensitivity probe standardizes **all** coordinates within each training fold and fits a linear Cox model with an L2 penalty. Its dimension before Cox is the global vector size in the table above; no PCA or nonlinear survival network is used. L2 regularization is necessary because each vector has more coordinates than the roughly 333 patients in an outer training fold. The penalty is selected with the existing three inner folds from {0.01, 0.1, 1, 10}, using Uno C at 36 months. An Efron-tie partial likelihood is optimized with L-BFGS; survival curves use a Breslow baseline. All scaling and penalty selection happen inside the training folds.

Four paired conditions use the same 416 patients and locked five-by-five outer folds:

| CT source | PCA32 Cox | Full embedding L2 Cox |
| --- | --- | --- |
| Original DICOM series | `outputs/lung1_e1_frozen_cox_four_models/` | `outputs/lung1_e1_frozen_cox_original_ridge/` |
| GTV preserving thorax HU crop | `outputs/lung1_e1_frozen_cox_crop_pca32/` | `outputs/lung1_e1_frozen_cox_crop_ridge/` |

Comparing rows at a fixed probe measures the input crop effect. Comparing columns at a fixed CT source measures the PCA versus full embedding effect. The existing original PCA32 run used a smaller penalty grid {0.01, 0.1, 1}; that difference should be kept in mind when interpreting its comparison with full ridge. An elastic-net Cox model could be tested later if sparse coordinate selection is of interest, though correlated encoder features may make selected coordinates unstable across folds.

All four runs are complete. Embeddings and results are under `outputs/`, with extraction and Cox logs under `logs/lung1_e1_crop/`. Every completed Cox outer fold was saved so a restarted run could resume. See the [results report](lung1_e1_crop_ridge_results_report.md) for the four-condition comparison.

The CT-CLIP checkpoint was obtained from a public mirror and has not been verified against the authors' gated checkpoint; this provenance limitation also applies to the crop sensitivity run.
