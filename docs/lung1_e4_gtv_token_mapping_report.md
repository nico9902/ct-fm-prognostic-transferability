# E4: GTV mapping onto cached frozen token grids

## Scope

The first E4 experiment will use frozen cached tokens and regional linear Cox probes. This audit maps the existing 1 mm LUNG1 GTV segmentation onto the **original CT** token cache for all 416 eligible patients and four foundation models. It does not fit a survival model.

The reproducible mapping is in [`scripts/audit_lung1_e4_gtv_token_mapping.py`](../scripts/audit_lung1_e4_gtv_token_mapping.py). Patient-level measurements are in `outputs/lung1_e4_geometry/patient_model_geometry.csv`; flattened per-token GTV footprint fractions are in `outputs/lung1_e4_geometry/weights/<model>/<patient_id>.npy`. Each array has the same row order as that patient's saved token array. The parallel `input_sample_weights/` arrays record the nearest-neighbor mask sampled onto the model input grid.

## Mapping method

1. Reconstruct the source DICOM CT geometry from the cached crop metadata, then place the cached GTV mask in physical space. The reconstruction was checked against three original DICOM series: source size, spacing, origin, and direction matched exactly.
2. Reproduce each model's input orientation, resampling, crop/pad, and spatial axis order. COLIPRI uses the official TorchIO image geometry transform. TANGERINE uses the 256³ resampling geometry. Rad-JEPA uses RAS orientation and a 32×256×256 grid. CT-CLIP uses LPS orientation, its resize geometry, and the 240×480×480 center crop/pad.
3. Assign the centers of 1 mm GTV voxels inside each final input field to nominal saved-token blocks. Divide the assigned 1 mm³ volume by block physical volume to obtain a **geometric GTV footprint fraction** per token. The saved tokens are pooled from the model's native token grid in the same axis order.
4. Separately, nearest-neighbor sample the mask onto the input voxel grid and count its occupancy in each block. This is a check on sampling sensitivity; the footprint fractions are the preferred regional weights.

The footprint gives nominal geometric overlap. It does **not** mean a token feature contains only that tissue: encoder receptive fields and attention mix information across locations. The blocks are a geometric approximation to saved-token positions, rather than an exact receptive-field calculation.

## Cohort results

| Model | Saved grid | Median nominal token span (mm; model axis order) | Median tokens touched by GTV | Patients with no token ≥10% GTV | Patients with no majority-GTV token | GTV field loss below 95% | Empty GTV after input-mask sampling |
|---|---:|---:|---:|---:|---:|---:|---:|
| COLIPRI | 8×8×8 | 48.0×48.0×48.0 (XYZ) | 7 | 156/416 | 362/416 | 1/416 | 0/416 |
| TANGERINE | 8×8×8 | 46.5×62.5×62.5 (ZYX) | 6 | 193/416 | 395/416 | 0/416 | 0/416 |
| Rad-JEPA-3D | 4×8×8 | 95.2×62.6×62.6 (ZYX) | 4 | 263/416 | 408/416 | 0/416 | 2/416 |
| CT-CLIP | 8×8×8 | 45.0×45.1×45.1 (ZYX) | 8 | 137/416 | 348/416 | 1/416 | 0/416 |

All 416 GTVs have a nonzero physical footprint on at least one saved token for every model. Median maximum GTV footprint fraction in any token is 0.160 (COLIPRI), 0.116 (TANGERINE), 0.064 (Rad-JEPA-3D), and 0.193 (CT-CLIP). Thus, even the most tumor-rich token is usually mostly other tissue.

**Edge cases.** LUNG1-038 has 93.1% GTV-center retention in COLIPRI's input field and 79.4% in CT-CLIP's; all other losses are below 1%. Rad-JEPA-3D retains the physical GTV field for LUNG1-108 and LUNG1-192, but nearest-neighbor sampling onto its 32-slice input leaves zero positive mask voxels. Their physical footprints still touch two and one saved token, respectively. The audit should therefore use physical footprint weights and carry a field-retention flag into E4 sensitivity analysis.

## Implications for the first E4 Cox experiment

- A tumor-weighted token pool is feasible for every patient and model. Use each token's GTV footprint fraction as its pooling weight, normalize weights within each patient, and report the effective number of contributing tokens. Avoid requiring a token to be majority GTV; that would discard most patients.
- Define the outside-tumor comparison using token blocks with **zero GTV footprint**. Weighting every token by `1 − GTV fraction` would put most tumor-overlapping tokens into both pools, because their GTV fractions are usually small. Report how many eligible blocks remain for each patient.
- A saved token spans roughly 45–95 mm along each axis. The proposed 0–5, 5–10, and 10–20 mm peritumoral rings cannot be cleanly separated on these saved grids. If narrow-ring attribution is required, cache finer native tokens and check their geometry; even then, encoder receptive fields remain broader than the nominal patch.
- Keep tumor, broader peritumoral, outside-tumor, and all-token pools as exploratory **spatial probes**, with identical Cox fitting and evaluation across models. A difference in probe performance will indicate where useful token *positions* occur, not prove that the feature was derived exclusively from that anatomic region.
- Predefine a sensitivity check that excludes LUNG1-038 for COLIPRI and CT-CLIP, plus a check for the two Rad-JEPA-3D input-sampling edge cases. The main geometric footprint analysis can retain them with the measured coverage reported.

Validation: 1,664 patient-model mappings completed. Every saved footprint and sampled-mask array has the expected token count, finite values in [0, 1], and statistics consistent with the patient-level file.
