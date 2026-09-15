#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TAG="${TAG:-pilot_1460_balanced}"
export EXPECTED_PATIENTS="${EXPECTED_PATIENTS:-1460}"
export EXPECTED_PER_CLASS="${EXPECTED_PER_CLASS:-730}"

bash "${ROOT_DIR}/scripts/run_server_build_token_transitions_pilot_2000.sh"
