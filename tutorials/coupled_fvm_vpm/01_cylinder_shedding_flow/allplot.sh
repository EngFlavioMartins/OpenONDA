#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/postprocess.py
python assets/plot_cylinder.py --format "${1:-png}"
python assets/compare_reference.py --reference "${REFERENCE_GRID:-medium}" --if-available --format "${1:-png}"
