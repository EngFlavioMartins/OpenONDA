#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_plate_polar.py --format "${1:-png}"
python assets/plot_plate_staticvsmoving.py --format "${1:-png}"
python assets/plot_plate_spanwise.py --format "${1:-png}"
python assets/plot_flat_plate_kelvin.py --format "${1:-png}"
python assets/plot_plate_velocity.py --format "${1:-png}"
python assets/plot_plate_impulse.py --format "${1:-png}"
python assets/render_flat_plate.py --format "${1:-png}"
