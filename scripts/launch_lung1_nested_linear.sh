#!/usr/bin/env bash
# Frozen linear Cox (PCA16 and PCA32) with the common E1 inner folds; repetition 0 is used
# by docs/lung1_sweep_validation_protocol.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/lung1_nested
for c in 16 32; do
  .venv/bin/python -u scripts/run_lung1_frozen_cox_e1.py --components $c  --embeddings colipri=outputs/lung1_e1_embeddings/colipri/original_ct/embedding_manifest.csv --embeddings tangerine=outputs/lung1_e1_embeddings/tangerine/original_ct/embedding_manifest.csv --embeddings radjepa=outputs/lung1_e1_embeddings/radjepa/original_ct/embedding_manifest.csv --embeddings ctclip=outputs/lung1_e1_embeddings/ctclip/original_ct/embedding_manifest.csv --embeddings dinov3=outputs/lung1_e1_embeddings/dinov3/original_ct/embedding_manifest.csv --embeddings pai=outputs/lung1_e1_embeddings/pai/original_ct/embedding_manifest.csv --embeddings hosny=outputs/lung1_e1_embeddings/hosny/original_ct/embedding_manifest.csv \
    --output-dir outputs/lung1_nested/linear_pca$c > logs/lung1_nested/linear_pca$c.log 2>&1
  echo "DONE pca$c $(date -Is)"
done
