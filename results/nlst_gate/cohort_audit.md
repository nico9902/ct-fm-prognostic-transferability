# Cohort audit

User approved the verified cached cohort after inspection: the active snapshot is pilot_1000_balanced, not a complete 2,000-patient COLIPRI cohort. The manifest contains 1,000 patients, 3,000 selected scans, and 500 cancer cases. The matched cache contains 999 patients, 2984 scans, and 1984 consecutive pairs. Selected scans and usable cached scans are counted separately.

Relative time uses scr_days0/1/2 via scan_day_from_randomization. Manifest study dates are deidentified; clinical intervals do not use their apparent calendar differences. Patients with only T0 or T0+T1 are retained in the audit; only observed consecutive pairs enter prognostic comparisons. One selected patient has no cached feature and no gate split.

Significant-change supervision is unavailable. sct_ab_gwth, sct_ab_invg, and sct_ab_preexist are lesion-record fields without a verified local scan-level coding rule. any_significant_abnormality and comparison_suspicious are current findings, not change labels.

The cohort is cancer-enriched; its prevalence and AUPRC do not estimate NLST population performance.

```text
   scope                                  metric       value
selected                                patients 1000.000000
selected                                   scans 3000.000000
selected                   patients_with_1_scans    0.000000
selected                   patients_with_2_scans    0.000000
selected                   patients_with_3_scans 1000.000000
selected                        pattern_T0+T1+T2 1000.000000
selected         patients_developing_lung_cancer  500.000000
selected                       cancer_prevalence    0.500000
selected     diagnosis_days_from_randomization_n  500.000000
selected diagnosis_days_from_randomization_count  500.000000
selected  diagnosis_days_from_randomization_mean 1480.224000
selected   diagnosis_days_from_randomization_std  529.307554
selected   diagnosis_days_from_randomization_min  509.000000
selected   diagnosis_days_from_randomization_25%  960.000000
selected   diagnosis_days_from_randomization_50% 1479.500000
selected   diagnosis_days_from_randomization_75% 1946.000000
selected   diagnosis_days_from_randomization_max 2618.000000
selected        time_to_cancer_after_scan_days_n 1496.000000
selected    time_to_cancer_after_scan_days_count 1496.000000
selected     time_to_cancer_after_scan_days_mean 1098.454545
selected      time_to_cancer_after_scan_days_std  608.419911
selected      time_to_cancer_after_scan_days_min    7.000000
selected      time_to_cancer_after_scan_days_25%  621.000000
selected      time_to_cancer_after_scan_days_50% 1075.500000
selected      time_to_cancer_after_scan_days_75% 1567.000000
selected      time_to_cancer_after_scan_days_max 2610.000000
  cached                                patients  999.000000
  cached                                   scans 2984.000000
  cached                   patients_with_1_scans    0.000000
  cached                   patients_with_2_scans   13.000000
  cached                   patients_with_3_scans  986.000000
  cached                        pattern_T0+T1+T2  986.000000
  cached                           pattern_T0+T1    9.000000
  cached                           pattern_T1+T2    3.000000
  cached                           pattern_T0+T2    1.000000
  cached         patients_developing_lung_cancer  499.000000
  cached                       cancer_prevalence    0.499499
  cached     diagnosis_days_from_randomization_n  499.000000
  cached diagnosis_days_from_randomization_count  499.000000
  cached  diagnosis_days_from_randomization_mean 1480.106212
  cached   diagnosis_days_from_randomization_std  529.832161
  cached   diagnosis_days_from_randomization_min  509.000000
  cached   diagnosis_days_from_randomization_25%  960.000000
  cached   diagnosis_days_from_randomization_50% 1479.000000
  cached   diagnosis_days_from_randomization_75% 1947.000000
  cached   diagnosis_days_from_randomization_max 2618.000000
  cached        time_to_cancer_after_scan_days_n 1483.000000
  cached    time_to_cancer_after_scan_days_count 1483.000000
  cached     time_to_cancer_after_scan_days_mean 1095.925152
  cached      time_to_cancer_after_scan_days_std  606.864610
  cached      time_to_cancer_after_scan_days_min    7.000000
  cached      time_to_cancer_after_scan_days_25%  618.500000
  cached      time_to_cancer_after_scan_days_50% 1071.000000
  cached      time_to_cancer_after_scan_days_75% 1565.000000
  cached      time_to_cancer_after_scan_days_max 2534.000000
  cached                      usable_pairs_T0_T1  995.000000
  cached                      usable_pairs_T1_T2  989.000000
  cached                        interscan_days_n 1984.000000
  cached                    interscan_days_count 1984.000000
  cached                     interscan_days_mean  369.990927
  cached                      interscan_days_std   48.413217
  cached                      interscan_days_min  125.000000
  cached                      interscan_days_25%  346.000000
  cached                      interscan_days_50%  364.000000
  cached                      interscan_days_75%  386.000000
  cached                      interscan_days_max  689.000000
  cached                          train_patients  599.000000
  cached        train_primary_future_cancer_1y_n 1186.000000
  cached train_primary_future_cancer_1y_positive  107.000000
  cached        train_primary_future_cancer_2y_n 1181.000000
  cached train_primary_future_cancer_2y_positive  256.000000
  cached        train_primary_future_cancer_3y_n 1173.000000
  cached train_primary_future_cancer_3y_positive  364.000000
  cached                            val_patients  200.000000
  cached          val_primary_future_cancer_1y_n  394.000000
  cached   val_primary_future_cancer_1y_positive   33.000000
  cached          val_primary_future_cancer_2y_n  391.000000
  cached   val_primary_future_cancer_2y_positive   81.000000
  cached          val_primary_future_cancer_3y_n  389.000000
  cached   val_primary_future_cancer_3y_positive  112.000000
  cached                           test_patients  200.000000
  cached         test_primary_future_cancer_1y_n  396.000000
  cached  test_primary_future_cancer_1y_positive   31.000000
  cached         test_primary_future_cancer_2y_n  396.000000
  cached  test_primary_future_cancer_2y_positive   75.000000
  cached         test_primary_future_cancer_3y_n  396.000000
  cached  test_primary_future_cancer_3y_positive  113.000000
  cached               significant_change_labels    0.000000
selected          requested_approximate_patients 2000.000000
```
