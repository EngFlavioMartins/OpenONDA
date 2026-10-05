#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_profile --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_comparison --format "${1:-both}"
