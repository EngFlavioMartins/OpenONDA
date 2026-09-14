#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_rotor_performance.py --format "${1:-png}"
python assets/plot_rotor_wake_planes.py --format "${1:-png}"
python assets/plot_rotor_loading_validation.py --format "${1:-png}"
python assets/plot_rotor_induction_validation.py --format "${1:-png}"
python assets/plot_rotor_streamwise.py --format "${1:-png}"
python assets/render_rotor_animation.py --fps 30
