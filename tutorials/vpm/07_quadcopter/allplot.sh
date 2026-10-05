#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_quadcopter_performance --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_quadcopter_wake --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_quadcopter_vorticity_history --format "${1:-both}"
