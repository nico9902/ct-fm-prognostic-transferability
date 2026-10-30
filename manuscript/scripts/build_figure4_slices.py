"""Draw the axial CT slices shown in the Input column of Figure 4 (rq3_design).

Axial slice through the GTV centroid of one LUNG1 patient, lung window, with the GTV
outline and, in the second image, the outline of the control region: the same GTV shape
translated within the axial plane to the mirror position across the body midline.
The control outline is illustrative; the analysis chose the control position per patient.
Usage: python scripts/build_figure4_slices.py [PATIENT_ID]
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/processed/lung1/thorax_hu"
OUT = ROOT / "manuscript/figures/figure4"
WINDOW = (-1250.0, 250.0)  # lung window, as in Figure 1
SIZE = 300
GTV_RGB = (200, 96, 63)  # testE in the figure palette
CONTROL_RGB = (138, 90, 158)  # extE

patient = sys.argv[1] if len(sys.argv) > 1 else "LUNG1-001"
ct = nib.as_closest_canonical(nib.load(SRC / patient / f"{patient}_volume.nii.gz")).get_fdata()
gtv = nib.as_closest_canonical(nib.load(SRC / patient / f"{patient}_gtv_mask.nii.gz")).get_fdata() > 0
cx, cy, cz = np.round(np.argwhere(gtv).mean(0)).astype(int)

# Radiological display: patient right on image left, anterior up.
show = lambda a: a.T[::-1, ::-1]
img = show(np.clip((ct[:, :, cz] - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0, 1))
mask = show(gtv[:, :, cz])
body = ndimage.binary_fill_holes(img > 0.45)
body = ndimage.binary_opening(body, iterations=3)
rows, cols = np.nonzero(body)
r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

# Control: GTV shape moved to the mirror column across the body midline, same row.
mid = (c0 + c1) / 2
gc = np.nonzero(mask)[1].mean()
control = ndimage.shift(mask.astype(float), (0, round(2 * (mid - gc))), order=0) > 0.5


def outline(m, width=3):
    return m & ~ndimage.binary_erosion(m, iterations=width) | (ndimage.binary_dilation(m, iterations=1) & ~m)


def render(overlays, name):
    rgb = np.repeat((img * 255).astype(np.uint8)[..., None], 3, axis=2)
    for m, colour in overlays:
        rgb[outline(m)] = colour
    crop = rgb[r0:r1, c0:c1]
    h, w = crop.shape[:2]
    n = max(h, w) + 8
    sq = np.zeros((n, n, 3), np.uint8)
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = crop
    Image.fromarray(sq).resize((SIZE, SIZE), Image.LANCZOS).save(OUT / name)


OUT.mkdir(parents=True, exist_ok=True)
render([(mask, GTV_RGB)], "axial_gtv.png")
render([(mask, GTV_RGB), (control, CONTROL_RGB)], "axial_gtv_control.png")
print(patient, "axial slice", cz, "->", OUT)
