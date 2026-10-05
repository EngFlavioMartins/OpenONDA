#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_vortex_ring_motion --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_vortex_ring_energy --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_vortex_ring_circulation --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_vortex_ring_scenes --format "${1:-both}"
