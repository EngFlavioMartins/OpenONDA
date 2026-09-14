#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/finalize_delta_wing_lineage.py
python assets/plot_delta_wing_forces.py --format "${1:-png}"
python assets/plot_delta_wing_circulation_history.py --format "${1:-png}"
python assets/plot_delta_wing_wake.py --format "${1:-png}"
python assets/render_delta_wing_gif.py --fps 30
