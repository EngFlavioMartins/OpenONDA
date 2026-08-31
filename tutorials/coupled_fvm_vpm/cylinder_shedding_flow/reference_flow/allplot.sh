#!/usr/bin/env bash
# Rebuild grid-study statistics and figures from existing samples.
set -euo pipefail

cd "$(dirname "$0")"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/openonda-matplotlib-cache"
mkdir -p "$MPLCONFIGDIR" figures

python -u assets/postprocess.py
python -u assets/plot_grid_study.py
