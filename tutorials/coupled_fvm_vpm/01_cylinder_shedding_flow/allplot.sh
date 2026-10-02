#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf]
cd -- "$(dirname -- "$0")"

python assets/plot_cylinder_forces.py --format "${1:-both}"
python assets/plot_reference_forces.py --format "${1:-both}"
python assets/plot_reference_profiles.py --format "${1:-both}"
