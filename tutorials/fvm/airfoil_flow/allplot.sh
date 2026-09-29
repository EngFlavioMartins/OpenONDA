#!/bin/bash
set -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_forces.py --format "${1:-png}"
python assets/plot_surface_cp.py --format "${1:-png}"
python assets/plot_velocity.py --format "${1:-png}"
