#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_profile.py --format "${1:-png}"
python assets/plot_comparison.py --format "${1:-png}"
