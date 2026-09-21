# Completed frozen COLIPRI NLST results

These results measure clinical decodability of pretrained representations. COLIPRI used NLST images through vision-only MAE pretraining; this is not fully independent external generalization.

Attention uses the frozen official pretrained pooling module; mean/max use raw dense features. All encoders remain frozen. Five-seed variability is sample SD, not a confidence interval.

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.507 ± 0.007 | 0.542 ± 0.013 | 0.795 ± 0.008 | 0.739 ± 0.006 | 0.782 ± 0.002 |
| nodule_ge_4mm | 0.504 ± 0.006 | 0.528 ± 0.014 | 0.788 ± 0.007 | 0.732 ± 0.004 | 0.777 ± 0.004 |
| emphysema | 0.555 ± 0.002 | 0.700 ± 0.011 | 0.752 ± 0.002 | 0.733 ± 0.004 | 0.738 ± 0.002 |
| fibrosis_or_scarring | 0.554 ± 0.005 | 0.603 ± 0.018 | 0.679 ± 0.004 | 0.640 ± 0.004 | 0.672 ± 0.005 |

## AUPRC

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.371 ± 0.009 | 0.375 ± 0.009 | 0.666 ± 0.009 | 0.608 ± 0.006 | 0.635 ± 0.005 |
| nodule_ge_4mm | 0.363 ± 0.007 | 0.361 ± 0.009 | 0.665 ± 0.009 | 0.613 ± 0.007 | 0.629 ± 0.007 |
| emphysema | 0.437 ± 0.003 | 0.612 ± 0.009 | 0.669 ± 0.005 | 0.642 ± 0.004 | 0.659 ± 0.006 |
| fibrosis_or_scarring | 0.345 ± 0.005 | 0.406 ± 0.023 | 0.454 ± 0.009 | 0.425 ± 0.004 | 0.455 ± 0.008 |

## MCC

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.024 ± 0.011 | 0.034 ± 0.014 | 0.461 ± 0.016 | 0.351 ± 0.013 | 0.414 ± 0.012 |
| nodule_ge_4mm | 0.014 ± 0.006 | 0.005 ± 0.024 | 0.459 ± 0.013 | 0.334 ± 0.008 | 0.426 ± 0.015 |
| emphysema | 0.088 ± 0.011 | 0.264 ± 0.025 | 0.392 ± 0.010 | 0.345 ± 0.005 | 0.354 ± 0.012 |
| fibrosis_or_scarring | 0.090 ± 0.009 | 0.132 ± 0.018 | 0.240 ± 0.012 | 0.171 ± 0.010 | 0.233 ± 0.009 |

## ACCURACY

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.512 ± 0.005 | 0.521 ± 0.006 | 0.754 ± 0.008 | 0.696 ± 0.007 | 0.730 ± 0.006 |
| nodule_ge_4mm | 0.508 ± 0.009 | 0.506 ± 0.011 | 0.752 ± 0.007 | 0.686 ± 0.003 | 0.736 ± 0.007 |
| emphysema | 0.547 ± 0.004 | 0.642 ± 0.014 | 0.704 ± 0.005 | 0.680 ± 0.002 | 0.683 ± 0.007 |
| fibrosis_or_scarring | 0.567 ± 0.003 | 0.582 ± 0.013 | 0.625 ± 0.006 | 0.603 ± 0.007 | 0.624 ± 0.007 |

## F1

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.424 ± 0.011 | 0.426 ± 0.010 | 0.652 ± 0.009 | 0.590 ± 0.007 | 0.623 ± 0.007 |
| nodule_ge_4mm | 0.414 ± 0.004 | 0.408 ± 0.014 | 0.651 ± 0.007 | 0.579 ± 0.006 | 0.631 ± 0.009 |
| emphysema | 0.480 ± 0.008 | 0.566 ± 0.012 | 0.641 ± 0.006 | 0.614 ± 0.004 | 0.622 ± 0.005 |
| fibrosis_or_scarring | 0.412 ± 0.008 | 0.440 ± 0.010 | 0.509 ± 0.008 | 0.462 ± 0.005 | 0.504 ± 0.006 |

## PRECISION

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.361 ± 0.005 | 0.367 ± 0.007 | 0.646 ± 0.013 | 0.559 ± 0.009 | 0.610 ± 0.009 |
| nodule_ge_4mm | 0.352 ± 0.003 | 0.348 ± 0.011 | 0.634 ± 0.014 | 0.539 ± 0.003 | 0.610 ± 0.009 |
| emphysema | 0.435 ± 0.005 | 0.537 ± 0.017 | 0.608 ± 0.007 | 0.581 ± 0.003 | 0.582 ± 0.010 |
| fibrosis_or_scarring | 0.349 ± 0.004 | 0.369 ± 0.011 | 0.420 ± 0.006 | 0.390 ± 0.006 | 0.418 ± 0.006 |

## RECALL

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.514 ± 0.025 | 0.507 ± 0.014 | 0.657 ± 0.011 | 0.625 ± 0.005 | 0.637 ± 0.009 |
| nodule_ge_4mm | 0.504 ± 0.016 | 0.493 ± 0.019 | 0.669 ± 0.011 | 0.626 ± 0.010 | 0.653 ± 0.011 |
| emphysema | 0.536 ± 0.014 | 0.599 ± 0.011 | 0.679 ± 0.006 | 0.652 ± 0.011 | 0.667 ± 0.004 |
| fibrosis_or_scarring | 0.502 ± 0.016 | 0.546 ± 0.019 | 0.644 ± 0.013 | 0.566 ± 0.007 | 0.633 ± 0.014 |

## SPECIFICITY

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | 0.511 ± 0.017 | 0.529 ± 0.004 | 0.806 ± 0.011 | 0.735 ± 0.009 | 0.780 ± 0.007 |
| nodule_ge_4mm | 0.510 ± 0.021 | 0.513 ± 0.009 | 0.796 ± 0.013 | 0.718 ± 0.002 | 0.780 ± 0.006 |
| emphysema | 0.555 ± 0.007 | 0.669 ± 0.021 | 0.719 ± 0.007 | 0.698 ± 0.007 | 0.693 ± 0.013 |
| fibrosis_or_scarring | 0.595 ± 0.008 | 0.597 ± 0.022 | 0.616 ± 0.009 | 0.619 ± 0.010 | 0.620 ± 0.013 |

## TN

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | nan | 205.200 ± 1.483 | 312.600 ± 4.219 | 285.000 ± 3.391 | 302.800 ± 2.864 |
| nodule_ge_4mm | nan | 200.400 ± 3.362 | 311.200 ± 5.263 | 280.600 ± 0.894 | 304.800 ± 2.490 |
| emphysema | nan | 243.400 ± 7.570 | 261.800 ± 2.683 | 254.200 ± 2.683 | 252.400 ± 4.722 |
| fibrosis_or_scarring | nan | 249.000 ± 9.327 | 257.000 ± 3.674 | 258.000 ± 4.183 | 258.400 ± 5.459 |

## FP

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | nan | 182.800 ± 1.483 | 75.400 ± 4.219 | 103.000 ± 3.391 | 85.200 ± 2.864 |
| nodule_ge_4mm | nan | 190.600 ± 3.362 | 79.800 ± 5.263 | 110.400 ± 0.894 | 86.200 ± 2.490 |
| emphysema | nan | 120.600 ± 7.570 | 102.200 ± 2.683 | 109.800 ± 2.683 | 111.600 ± 4.722 |
| fibrosis_or_scarring | nan | 168.000 ± 9.327 | 160.000 ± 3.674 | 159.000 ± 4.183 | 158.600 ± 5.459 |

## FN

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | nan | 103.000 ± 3.000 | 71.600 ± 2.302 | 78.400 ± 1.140 | 75.800 ± 1.789 |
| nodule_ge_4mm | nan | 104.400 ± 3.912 | 68.200 ± 2.280 | 77.000 ± 2.000 | 71.400 ± 2.191 |
| emphysema | nan | 93.400 ± 2.510 | 74.800 ± 1.483 | 81.000 ± 2.449 | 77.600 ± 0.894 |
| fibrosis_or_scarring | nan | 81.800 ± 3.421 | 64.000 ± 2.345 | 78.200 ± 1.304 | 66.000 ± 2.550 |

## TP

| Target | Rad-JEPA | TANGERINE | COLIPRI mean | COLIPRI max | COLIPRI attention |
| --- | --- | --- | --- | --- | --- |
| isolation_suspicious | nan | 106.000 ± 3.000 | 137.400 ± 2.302 | 130.600 ± 1.140 | 133.200 ± 1.789 |
| nodule_ge_4mm | nan | 101.600 ± 3.912 | 137.800 ± 2.280 | 129.000 ± 2.000 | 134.600 ± 2.191 |
| emphysema | nan | 139.600 ± 2.510 | 158.200 ± 1.483 | 152.000 ± 2.449 | 155.400 ± 0.894 |
| fibrosis_or_scarring | nan | 98.200 ± 3.421 | 116.000 ± 2.345 | 101.800 ± 1.304 | 114.000 ± 2.550 |

## Smoke test

```json

{
  "original_ct_shape": [
    1,
    512,
    512,
    124
  ],
  "original_spacing_mm": [
    0.566406,
    0.566406,
    2.0000000081300815
  ],
  "resampled_ct_shape": [
    1,
    144,
    144,
    124
  ],
  "colipri_input_shape": [
    1,
    1,
    192,
    192,
    192
  ],
  "dense_grid_shape": [
    1,
    864,
    24,
    24,
    24
  ],
  "dense_token_shape": [
    1,
    13824,
    864
  ],
  "pooled_embedding_shape": {
    "mean": [
      1,
      864
    ],
    "max": [
      1,
      864
    ],
    "attention": [
      1,
      768
    ]
  },
  "smoke_seconds": 7.667955467943102,
  "official_attention_equivalence": true
}

```

Rad-JEPA uses the historical best pure linear representation per target selected by test AUROC, a descriptive retrospective baseline. Interpret suspicion/nodule gains alongside AUPRC and MCC, not AUROC alone. This scan-level test does not demonstrate spatial localization. Preprocessing differs according to each model’s intended contract.
