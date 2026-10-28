#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_rotor_performance --format "${1:-both}"
python -m assets.plot_rotor_wake_planes --format "${1:-both}"
python -m assets.plot_rotor_loading_validation --format "${1:-both}"
python -m assets.plot_rotor_streamwise --format "${1:-both}"
python -m assets.render_rotor_animation --fps 30
python -m assets.render_rotor_wake_animation
