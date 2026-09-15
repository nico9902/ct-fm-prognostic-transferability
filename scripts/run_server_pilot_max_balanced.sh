#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TAG="${TAG:-pilot_1460_balanced}"
export N_PATIENTS="${N_PATIENTS:-1460}"

bash "${ROOT_DIR}/scripts/run_server_pilot_2000_balanced.sh"
