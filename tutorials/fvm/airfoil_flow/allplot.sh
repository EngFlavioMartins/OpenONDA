#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_forces --format "${1:-both}"
python -m assets.plot_surface_cp --format "${1:-both}"
python -m assets.plot_velocity --format "${1:-both}"
