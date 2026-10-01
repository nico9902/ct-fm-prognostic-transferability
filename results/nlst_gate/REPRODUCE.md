# Reproduce the gate

From the repository root, using its existing `.venv`:

```bash
PYTHONPATH=src:scripts .venv/bin/python tests/test_nlst_gate.py
PYTHONPATH=src:scripts .venv/bin/python scripts/run_nlst_gate.py --audit-only
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/nlst-gate-mpl .venv/bin/python scripts/run_nlst_gate.py --config results/nlst_gate/configs/gate.yaml
PYTHONPATH=src:scripts .venv/bin/python scripts/validate_nlst_gate_outputs.py --log-wandb
```

The full run restores finished fits on resume, guarded by source/config/data SHA256 hashes. To run a genuinely new experiment, copy the gate YAML and set a new `output_dir`. Do not silently alter an existing run. This completed run used a Tesla V100-PCIE-16GB, as recorded in environment.json. The device index can be changed to an available GPU; set CUDA_DEVICE_ORDER=PCI_BUS_ID to match nvidia-smi numbering. The script uses the visible GPU; CPU is supported but will be slower.

`--smoke` uses a separate output directory, one seed, two training epochs, and 20 bootstrap draws. Smoke results are pipeline checks, not scientific evidence. `--report-only` regenerates the analysis/report from saved predictions and states without training. This flag is useful after an interrupted analysis stage, and does not perform the training fingerprint check.

The active dataset is `pilot_1000_balanced` and the verified COLIPRI cache contains 999 patients. The user approved this scope. Existing 599/200/200 patient assignments are reused. No new split or dynamics model is trained. Only the disease projector and downstream heads are learned; COLIPRI stays frozen.

Global, full pre-pooling dense, and coarsened spatial token caches remain at their existing paths. `feature_cache_manifest.csv` explicitly maps each cached scan to those arrays and to its clinical metadata. Mean-pooling equivalence is verified in the runner. The same `[4,128]` query state is frozen across all disease arms within a seed.

`configs/gate.yaml` is the authoritative runnable configuration. Per-experiment JSON files document exact inputs, architecture, seeds, tasks, oracle status, and shared configuration. `configs/*_scaler.npz`, `token_scaler.npz`, PCA fits, best checkpoints, CSV training histories, and per-scan/per-pair predictions preserve preprocessing and evaluation details.

Training uses offline W&B and CSV logging; final consolidated train/val/test metrics and bootstrap comparisons are logged by the validator with `--log-wandb`. Cloud synchronization is not required. Full source fingerprints and cache-checkpoint identity are in `fingerprints.json`; runtime versions and tensor shapes are in `environment.json`.
