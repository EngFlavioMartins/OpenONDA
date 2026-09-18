#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_delta_wing_forces.py --format "${1:-png}"
python assets/plot_delta_wing_force_cycles.py --format "${1:-png}"
python assets/plot_delta_wing_circulation_history.py --format "${1:-png}"
python assets/plot_delta_wing_wake_streamwise.py --format "${1:-png}"
python assets/plot_delta_wing_wake_vertical.py --format "${1:-png}"
python assets/postprocess.py render-gif