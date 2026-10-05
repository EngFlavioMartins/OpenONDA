#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_delta_wing_forces --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_delta_wing_force_cycles --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_delta_wing_circulation_history --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_delta_wing_wake_streamwise --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_delta_wing_wake_vertical --format "${1:-both}"
python -m openonda.tutorial_runner . assets.postprocess render-gif
