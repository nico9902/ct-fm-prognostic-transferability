# LUNG1 pilot using the LUNA16 preprocessing recipe

## Method

`scripts/preprocess_lung1_luna_style.py` adapts the LUNA16 workflow in `/home/domenico/lung_ct_3d2d_synthesis/src/prs/luna16_preprocessing.py` to LUNG1 DICOM series. For each region-ready LUNG1 patient it:

1. decodes the selected DICOM CT in HU;
2. resamples to 1 mm isotropic spacing using linear interpolation;
3. clips to `[-1200, 600]` HU and stores uint8 intensities in `[0, 255]`;
4. infers lungs with `lungmask` R231, dilates the lung mask by 10 voxels, zeros intensities outside that dilated mask, then crops to the lung bounding box plus 10 voxels;
5. maps the TCIA primary-neoplasm DICOM SEG into CT coordinates with nearest-neighbor interpolation, applies the same crop, and saves an aligned GTV mask;
6. writes a preview and metadata with the fraction of GTV retained by the crop and lung masking.

This uses the other project's Python environment because it already contains `lungmask` and its cached R231 checkpoint. The output is independent of that project.

## Pilot command

```bash
/home/domenico/lung_ct_3d2d_synthesis/myenv/bin/python -u \
  scripts/preprocess_lung1_luna_style.py \
  --patient-ids LUNG1-001 LUNG1-023 LUNG1-282 --force-cpu
```

Outputs are under `data/processed/lung1/luna_style/<patient_id>/`, with `<patient_id>_volume.nii.gz`, `<patient_id>_lung_mask.nii.gz`, `<patient_id>_gtv_mask.nii.gz`, `<patient_id>_metadata.json`, and `<patient_id>_preview.png`. `preprocessing_summary.csv` collects the pilot metadata.

## Initial observations

| Patient | Source CT | GTV retained inside crop | GTV retained after lung masking | Interpretation |
| --- | --- | ---: | ---: | --- |
| LUNG1-001 | 512 × 512 × 134; 0.977 × 0.977 × 3 mm | 100% | 93.5% | Tumor partly removed |
| LUNG1-023 | see metadata | see metadata | 56.2% | Large tumor substantially removed |

The output images demonstrate that the exact lung-masked LUNA recipe can erase NSCLC tumor tissue. The LUNG1-001 GTV volume on the resampled grid is 155.79 cm³ versus 156.32 cm³ in the source SEG, a 0.34% difference; the loss shown above occurs at the masking step. In LUNG1-023, the preview shows a large black region inside the red GTV contour after masking.

The masked CTs are a reproducible LUNA-style variant. For the proposal's **primary whole-chest foundation-model benchmark**, retain an unmasked HU volume and use each encoder's prescribed input processing. This preserves tumor, mediastinal, and other thoracic context needed to test the proposal's spatial hypotheses. The lung-masked variant can be a predefined sensitivity or regional-ablation input.

## Unmasked thorax bounding box

The script also provides `--mode thorax_hu`. It builds one 3-D bounding box from the union of both lung masks and the aligned GTV, adds a 10 mm margin on the 1 mm grid, and retains every CT voxel inside the box. Thus the central area between the lungs is preserved. The saved CT remains float32 HU; the lung-window mapping is used only for the preview. The GTV is included in the box so that a peripheral or mediastinal extension cannot be cropped away. This GTV-informed field of view should be recorded when comparing with an annotation-free preprocessing policy.

```bash
/home/domenico/lung_ct_3d2d_synthesis/myenv/bin/python -u \
  scripts/preprocess_lung1_luna_style.py \
  --mode thorax_hu --patient-ids LUNG1-001 LUNG1-023 --force-cpu
```

The default output is `data/processed/lung1/thorax_hu/`; it does not replace the LUNA-style pilot. Metadata records the GTV fraction retained in the crop, and processing fails if it is below 100%. Each foundation model still needs its own prescribed resampling, intensity scaling, and fixed input size after this crop.

In the LUNG1-023 verification run, the saved float32 HU crop retained all 659,853 resampled GTV voxels. Of these, 549,268 lie outside the saved lung mask; their CT intensities remain present in the new crop. The earlier lung-masked image retained only 56.2% of the GTV.
