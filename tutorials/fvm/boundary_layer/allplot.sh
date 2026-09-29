#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_blasius.py --format "${1:-png}"
python assets/plot_cf.py --format "${1:-png}"
