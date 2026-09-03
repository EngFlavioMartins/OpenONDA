#!/usr/bin/env bash
# Rebuild the Lamb--Oseen diagnostics and figures from existing samples.
set -euo pipefail

cd "$(dirname "$0")"
MODULE="tutorials.vpm.lamb_oseen_vortex"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/openonda-matplotlib-cache"
mkdir -p "$MPLCONFIGDIR" figures

python -m "${MODULE}.assets.postprocess" --extract-fields

python -m "${MODULE}.assets.plot_vortex_comparison" --format both
python -m "${MODULE}.assets.plot_dipole_comparison" --format both
python -m "${MODULE}.assets.plot_merging_comparison" --format both
python -m "${MODULE}.assets.plot_vortex_surface_fields" --format both
python -m "${MODULE}.assets.plot_lamboseen_energy" --format both

python -m "${MODULE}.assets.postprocess" --manifest
