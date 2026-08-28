#!/usr/bin/env bash
# Rebuild the Lamb--Oseen diagnostics and figures from existing samples.
set -euo pipefail

cd "$(dirname "$0")"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/openonda-matplotlib-cache"
mkdir -p "$MPLCONFIGDIR" figures

python assets/postprocess.py --extract-fields

python assets/plot_vortex_comparison.py --format both
python assets/plot_dipole_comparison.py --format both
python assets/plot_merging_comparison.py --format both
python assets/plot_vortex_surface_fields.py --format both
python assets/plot_lamboseen_energy.py --format both

python assets/postprocess.py --manifest
