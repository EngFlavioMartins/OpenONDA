#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_rotor_performance --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_rotor_wake_planes --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_rotor_loading_validation --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_rotor_streamwise --format "${1:-both}"
python -m openonda.tutorial_runner . assets.render_rotor_animation --fps 30
