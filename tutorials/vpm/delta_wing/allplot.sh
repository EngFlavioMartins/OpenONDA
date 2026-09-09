#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python assets/plot_delta_wing_forces.py --format "${1:-png}"
python assets/plot_delta_wing_circulation_history.py --format "${1:-png}"
python assets/plot_delta_wing_wake.py --format "${1:-png}"
