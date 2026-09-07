#!/usr/bin/env bash
# Qualified spatial + temporal + iterative study, then statistics and plots.
set -euo pipefail

cd "$(dirname "$0")"

export MPLBACKEND=Agg
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-6}"
exec "${PYTHON:-python}" -u study.py "$@"
