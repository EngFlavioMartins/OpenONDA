#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_delta_wing_forces.py --format "${1:-both}"
python assets/plot_delta_wing_force_cycles.py --format "${1:-both}"
python assets/plot_delta_wing_circulation_history.py --format "${1:-both}"
python assets/plot_delta_wing_wake_streamwise.py --format "${1:-both}"
python assets/plot_delta_wing_wake_vertical.py --format "${1:-both}"
python assets/postprocess.py render-gif
