#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-pilot_2000_balanced}" GPU_ID="${GPU_ID:-1}" bash scripts/run_server_train_transformer_token_cls_z_only_gpu2.sh
