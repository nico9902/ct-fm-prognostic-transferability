# TANGERINE implementation and NLST compatibility audit

Inspected 17 September 2026. Official source: [3D-MAE-MedImaging](https://github.com/niccolo246/3D-MAE-MedImaging), commit `5bc575f593197097e4d68503bb94bb1c9bb4f1d3`. Checkpoint: [Zenodo record 18835750](https://zenodo.org/records/18835750), `tangerine.pth`, 1,362,743,936 bytes. Its published MD5, `d594de5ec0f59c55a1c20ec9f6ad3bef`, was verified after download. SHA256: `ebd9a60eb8cce3fbc05012dc89a0b075750e166ce4b432d1c75a0e1839702c6c`.

## Input and preprocessing

The [paper](https://www.nature.com/articles/s43856-025-01328-1), `model_card.md`, `patch_three_d.py`, and `datasets_three_d_fine.py` specify one-channel 256×256×256 CTs, in `[B,C,D,H,W]` order. SimpleITK returns arrays in `[D,H,W]` order; no additional orientation permutation is used for the standard NIfTI pipeline. The old ILA orientation option is not appropriate for this experiment.

The official utility resamples to 256³ using SimpleITK B-spline interpolation. New spacing is `original_size * original_spacing / 256`; origin and direction are preserved. Resampled intensities are clipped to [-1200,800] HU and scaled with `(HU+1200)/2000`. Inference has no augmentation or MAE masking. The embedding exporter itself expects volumes already resized to the model input size; it does not perform this resampling automatically.

The prior Rad-JEPA arrays `[1,D,256,256]` have irreversible foreground cropping and [-1000,400] HU clipping. Multiplying by 1400 and subtracting 1000 only recovers clipped HU estimates. Exact official TANGERINE preprocessing requires the original HU CT of the same selected series. The primary experiment uses those originals, preserving all prior scan selections, diagnostic slice filters, duplicate-plane removal, exclusions, labels, and patient splits. A separate `prepared_approximation` option supports the existing arrays with this limitation explicitly identified.

## Loading the encoder

Instantiate `models_vit.vit_large_patch16_yo(img_size=256, num_classes=0, global_pool=True, drop_path_rate=0)`. The released checkpoint includes both encoder and decoder tensors, despite the README's encoder-weight wording. The official downstream scripts select the checkpoint's `model` or `model_state` mapping and retain matching keys with matching tensor shapes. No positional interpolation is necessary at the released 256³ size.

All 292 tensors used by the downstream encoder load. Unused MAE decoder and final MAE `norm` tensors are excluded. The only missing downstream tensors are `fc_norm.weight` and `fc_norm.bias`: the official extractor consequently retains the new LayerNorm's deterministic defaults, weight=1, bias=0, epsilon=1e-6. This normalization is not a pretrained learned layer and is also kept frozen.

The experiment additionally rejects missing encoder tensors instead of silently keeping random initialization. The encoder and normalization use `requires_grad_(False)`, evaluation mode, and `torch.inference_mode()` throughout.

## Recommended whole-volume embedding

The [official embedding exporter](https://github.com/niccolo246/3D-MAE-MedImaging/blob/5bc575f593197097e4d68503bb94bb1c9bb4f1d3/mains_predict/extract_encoder_embeddings.py) has:

```python
global_pool = True
embedding_type = "pre_logits"  # default
cls = tokens[:, 0]
mean_patch = tokens[:, 1:].mean(dim=1)
concat = torch.cat((cls, mean_patch), dim=1)
embedding = model.fc_norm(concat)
```

Thus the recommended representation is **LayerNorm-normalized CLS + mean-patch concatenation**, 2,048 dimensions. `class`, `pooled`, and raw `concat` are available alternatives; they are not the exporter's default. `normalized_concat` aliases `pre_logits`. The standard downstream model's `forward_features` returns the raw concatenation, so calling it alone omits the final step recommended by the exporter.

## Smoke shapes

The actual checkpoint smoke test produced finite features:

| Tensor | Shape |
| --- | --- |
| Prior prepared CT | `[1,124,256,256]` |
| Official preprocessed CT | `[1,256,256,256]` |
| Batched encoder input | `[1,1,256,256,256]` |
| 16³ patch grid | `[16,16,16]` |
| Patch tokens | `[1,4096,1024]` |
| Tokens including CLS | `[1,4097,1024]` |
| CLS and mean-patch vectors, each | `[1,1024]` |
| Concatenation and normalized pre-logits | `[1,2048]` |
| Linear logits | `[1,1]` |
| MLP hidden features | `[1,128]` |
| MLP logits | `[1,1]` |

Patch and transformer-token shapes were observed with hooks. Both probe architectures were run on a cached feature to check their shapes.

## Potential NLST pretraining overlap

The model card's generic private-dataset description is less informative than the published paper. The paper identifies NLST-LSS as a major pretraining source. [Supplementary Data 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43856-025-01328-1/MediaObjects/43856_2025_1328_MOESM3_ESM.xlsx) lists 30,760 LSS series from 10,541 patients. ACRIN is described as a pretraining-unseen downstream dataset.

The [official NLST dictionary](https://www.cancerimagingarchive.net/wp-content/uploads/NLST_Database_Query_Tool_Data_Dictionary_2012_10_25.pdf) maps 1xxxxx PIDs to LSS and 2xxxxx to ACRIN. Our usable cohort contains 664 LSS patients / 1,988 scans and 335 ACRIN patients / 996 scans. In the fixed test split, 131 LSS patients / 391 scans are potentially exposed, and 69 ACRIN patients / 206 scans belong to the component reported as unseen. Ten DICOM PatientName headers per component agreed with this mapping.

Neither the checkpoint nor the dataset table provides an individual pretraining membership list. Exact patient/series overlap cannot be determined. Full-cohort performance must be interpreted as frozen-feature decodability in a potentially exposed cohort, with the matched ACRIN subset reported as a sensitivity analysis. ACRIN component membership alone is not a definitive patient-level guarantee against cross-dataset duplication.

Implementation: `scripts/run_tangerine_decodability.py`. Results and the scientific comparison are reported separately after extraction and all five-seed probes finish.
