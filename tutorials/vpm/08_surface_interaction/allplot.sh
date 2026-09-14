#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)

cd -- "$(dirname -- "$0")"
python assets/plot_qualification.py --format "${1:-png}"
