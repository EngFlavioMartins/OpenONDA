#!/usr/bin/env bash
# Rebuild the coupled-cylinder statistics and figures from existing samples.
set -euo pipefail

cd "$(dirname "$0")"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/openonda-matplotlib-cache"
mkdir -p "$MPLCONFIGDIR" figures

python assets/postprocess.py
python assets/plot_cylinder.py
