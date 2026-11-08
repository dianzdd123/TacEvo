#!/usr/bin/env bash
# Activate an explicitly prepared environment before starting this launcher.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python scripts/run_pipeline.py "$@"
