#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_plate_polar.py --format "${1:-both}"
python assets/plot_plate_staticvsmoving.py --format "${1:-both}"
python assets/plot_plate_spanwise.py --format "${1:-both}"
python assets/plot_flat_plate_kelvin.py --format "${1:-both}"
python assets/plot_plate_velocity.py --format "${1:-both}"
python assets/plot_plate_impulse.py --format "${1:-both}"
python assets/render_flat_plate.py --format "${1:-both}"

python assets/validate_results.py --pre-plot
