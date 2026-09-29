#!/bin/bash
set -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_forces.py --format "${1:-png}"
python assets/plot_wake.py --format "${1:-png}"
python assets/plot_fields.py --format "${1:-png}"
