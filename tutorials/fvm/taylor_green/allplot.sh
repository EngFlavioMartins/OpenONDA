#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_decay --history solution/history.csv --output "figures/taylor_green_decay.${1:-both}"
