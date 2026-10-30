#!/usr/bin/env bash
# Wait for the resumable input cache, then run the secondary encoder rungs.
set -euo pipefail
cd "$(dirname "$0")/.."
while [[ ! -f outputs/lung1_e2_inputs/COMPLETE ]]; do
  sleep 30
done
exec bash scripts/launch_lung1_e2_encoder_gpu.sh
