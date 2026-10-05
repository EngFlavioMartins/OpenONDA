#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_plate_polar --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_plate_staticvsmoving --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_plate_spanwise --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_plate_velocity --format "${1:-both}"
python -m openonda.tutorial_runner . assets.render_flat_plate --format "${1:-both}"
