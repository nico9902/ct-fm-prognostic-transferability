#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-pilot_2000_balanced}" bash scripts/run_server_extract_radjepa_tokens_pilot_500.sh
