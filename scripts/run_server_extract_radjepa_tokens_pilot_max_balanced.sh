#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TAG="${TAG:-pilot_1460_balanced}"

bash "${ROOT_DIR}/scripts/run_server_extract_radjepa_tokens_pilot_500.sh"
