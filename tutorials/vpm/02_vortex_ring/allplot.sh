#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_vortex_ring_motion.py --format "${1:-both}"
python assets/plot_vortex_ring_energy.py --format "${1:-both}"
python assets/plot_vortex_ring_circulation.py --format "${1:-both}"
python assets/plot_vortex_ring_scenes.py --available --format "${1:-both}"
