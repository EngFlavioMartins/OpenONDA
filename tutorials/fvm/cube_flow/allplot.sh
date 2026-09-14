#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_forces.py --format "${1:-png}"
python assets/plot_vorticity.py --format "${1:-png}"
python assets/plot_wake.py --format "${1:-png}"
