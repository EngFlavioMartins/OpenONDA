#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_decay.py --history solution/history.csv --output "figures/taylor_green_decay.${1:-png}"
