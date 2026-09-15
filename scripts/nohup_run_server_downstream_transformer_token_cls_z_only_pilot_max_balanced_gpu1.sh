#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TAG="${TAG:-pilot_1460_balanced}"
export GPU_ID="${GPU_ID:-1}"

bash "${ROOT_DIR}/scripts/nohup_run_server_downstream_transformer_token_cls_z_only_multiseed_gpu2.sh"
