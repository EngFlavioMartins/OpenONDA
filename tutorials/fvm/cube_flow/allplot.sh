#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_forces --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_vorticity --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_wake --format "${1:-both}"
