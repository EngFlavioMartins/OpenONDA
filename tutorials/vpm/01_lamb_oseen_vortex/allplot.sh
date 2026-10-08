#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.postprocess --extract-fields
python -m assets.plot_vortex_comparison --format "${1:-both}"
python -m assets.plot_dipole_comparison --format "${1:-both}"
python -m assets.plot_merging_comparison --format "${1:-both}"
python -m assets.plot_vortex_surface_fields --format "${1:-both}"
python -m assets.plot_lamboseen_energy --format "${1:-both}"
python -m assets.plot_merging_snapshots --format "${1:-both}"
