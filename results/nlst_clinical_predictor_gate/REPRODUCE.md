# Reproduce the second gate

Use the existing repository `.venv` from the project root. The completed cohort from `results/nlst_gate` and existing COLIPRI caches must remain available.

```bash
PYTHONPATH=src:scripts .venv/bin/python tests/test_clinical_predictor_gate.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-81bd61f7-09c3-c49e-287d-9804bc6e9e01 PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/nlst-clinical-gate-mpl .venv/bin/python scripts/run_nlst_clinical_predictor_gate.py
PYTHONPATH=src:scripts .venv/bin/python scripts/validate_nlst_clinical_predictor_gate.py --log-wandb
```

A GPU UUID explicitly selects the A100 used for this run, avoiding CUDA index ambiguity. Substitute another available GPU if necessary. No additional packages or model/data downloads are required.

`configs/gate.yaml` is the authoritative configuration. All hyperparameters, loss-weight grids, patient splits, targets and seed choices are fixed before test evaluation. Per-candidate JSON files, train-only scaler statistics, validation-only oracle selection and predictor selection are retained. The test cohort is not changed.

Primary predictors see only the current raw global embedding and a configured 365-day query. Time input is standardized using the saved training-only time scaler: `(365-time_mean)/time_scale`, which is zero here. Future state and labels are arguments to the loss, never to `forward`. The old checkpoint ensemble is reevaluated with the same query; the separately reproduced old realized-interval ensemble is a retrospective legacy reference and is excluded from prospective gate comparisons.

Residual output is transformed with train-delta mean/scale and added to current state. Direct output is transformed with train-future mean/scale. Both use the SAME old backbone class. The frozen oracle includes its train-future input scaler and three validation-selected risk heads. Its `.train()` remains in evaluation mode and its parameters are absent from the optimizer.

`--smoke` uses a separate directory, one seed, two epochs, a single clinical weight, and 20 bootstrap draws. It verifies the pipeline and carries no scientific interpretation. `--analysis-only` reloads saved predictions and oracle states to regenerate analysis without training. Resume uses fingerprints of code/config/inputs and refuses incompatible training artifacts; use a new `output_dir` for a new experiment.

All stages use offline W&B plus CSV histories. The final validator logs consolidated metrics/tables to a separate offline W&B run. Cloud synchronization is not needed. Large checkpoints and predicted embedding caches remain local and are ignored by Git, while tables, reports, configs and plots are available in the result directory.

This experiment does not implement a world model or the optional disease-projector follow-up.
