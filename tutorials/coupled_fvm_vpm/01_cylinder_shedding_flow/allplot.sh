#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_cylinder_forces.py --format "${1:-png}"
python assets/plot_reference_forces.py --format "${1:-png}"
python assets/plot_reference_profiles.py --format "${1:-png}"
