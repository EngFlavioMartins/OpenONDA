#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_blasius --format "${1:-both}"
python -m assets.plot_cf --format "${1:-both}"
