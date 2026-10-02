#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_rotor_performance.py --format "${1:-both}"
python assets/plot_rotor_wake_planes.py --format "${1:-both}"
python assets/plot_rotor_loading_validation.py --format "${1:-both}"
python assets/plot_rotor_streamwise.py --format "${1:-both}"
python assets/render_rotor_animation.py --fps 30

python assets/validate_results.py --pre-plot
