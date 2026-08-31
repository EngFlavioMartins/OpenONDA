#!/usr/bin/env bash
# Run the coupled FVM--VPM cylinder-shedding case.
set -euo pipefail

cd "$(dirname "$0")"

rm -rf solution samples figures
mkdir -p solution samples figures

python -u setup.py

python assets/postprocess.py
python assets/plot_cylinder.py
