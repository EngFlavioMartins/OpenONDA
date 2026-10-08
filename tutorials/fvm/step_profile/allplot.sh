#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_profile --format "${1:-both}"
python -m assets.plot_comparison --format "${1:-both}"
