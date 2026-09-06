#!/usr/bin/env bash
# Rebuild grid-study statistics and figures from existing samples.
set -euo pipefail

cd "$(dirname "$0")"

export MPLBACKEND=Agg
exec "${PYTHON:-python}" -u study.py --report-only "$@"
