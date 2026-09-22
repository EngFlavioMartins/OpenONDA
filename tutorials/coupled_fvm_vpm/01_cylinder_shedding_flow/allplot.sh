#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] [--run-dir directory]
cd -- "$(dirname -- "$0")"

python assets/plot_campaign.py --format "${1:-png}" "${@:2}"
