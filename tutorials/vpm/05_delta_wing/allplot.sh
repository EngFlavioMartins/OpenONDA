#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"
python assets/plot_delta_wing.py --format "${1:-png}"
