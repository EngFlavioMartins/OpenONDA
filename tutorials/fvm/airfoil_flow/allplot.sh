#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_forces.py --format "${1:-both}"
python assets/plot_surface_cp.py --format "${1:-both}"
python assets/plot_velocity.py --format "${1:-both}"
