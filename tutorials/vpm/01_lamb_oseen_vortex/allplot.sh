#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/postprocess.py --extract-fields
python assets/plot_vortex_comparison.py --format "${1:-both}"
python assets/plot_dipole_comparison.py --format "${1:-both}"
python assets/plot_merging_comparison.py --format "${1:-both}"
python assets/plot_vortex_surface_fields.py --format "${1:-both}"
python assets/plot_lamboseen_energy.py --format "${1:-both}"
python assets/plot_merging_snapshots.py --format "${1:-both}"
