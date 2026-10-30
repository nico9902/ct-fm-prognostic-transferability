"""Extract the three CT slices shown in panel a of Figure 1 (study_overview).

Sagittal, coronal and axial slices through the GTV centroid of one LUNG1 patient,
lung window, padded to squares with a black background as in the reference figure.
Usage: python scripts/build_figure1_slices.py [PATIENT_ID]
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/processed/lung1/thorax_hu"
OUT = ROOT / "manuscript/figures/figure1"
WINDOW = (-1250.0, 250.0)  # lung window: level -500 HU, width 1500 HU
SIZE = 300

patient = sys.argv[1] if len(sys.argv) > 1 else "LUNG1-001"
ct = nib.as_closest_canonical(nib.load(SRC / patient / f"{patient}_volume.nii.gz")).get_fdata()  # RAS
gtv = nib.as_closest_canonical(nib.load(SRC / patient / f"{patient}_gtv_mask.nii.gz")).get_fdata() > 0
cx, cy, cz = np.round(np.argwhere(gtv).mean(0)).astype(int)
img = np.clip((ct - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0, 1)

# Radiological display (array axes are RAS): axial with patient right on image left
# and anterior up; coronal and sagittal with superior up and right/anterior on the left.
slices = {
    "sagittal": img[cx, :, :].T[::-1, ::-1],
    "coronal": img[:, cy, :].T[::-1, ::-1],
    "axial": img[:, :, cz].T[::-1, ::-1],
}
OUT.mkdir(parents=True, exist_ok=True)
for name, s in slices.items():
    h, w = s.shape
    n = max(h, w)
    sq = np.zeros((n, n))
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = s
    Image.fromarray((sq * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS).save(
        OUT / f"slice_{name}_square.png")
print(patient, "GTV centroid voxel", (cx, cy, cz), "->", OUT)
