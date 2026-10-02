#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/plot_decay.py --history solution/history.csv --output "figures/taylor_green_decay.${1:-both}"
