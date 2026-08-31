#!/usr/bin/env bash
# Run the three leapfrogging LES stabilization comparisons.
set -euo pipefail

cd "$(dirname "$0")"

rm -rf solution samples figures
mkdir -p solution samples figures

python -u setup.py --case leapfrog_les
python -u setup.py --case leapfrog_les_splitting
python -u setup.py --case leapfrog_les_splitting_remeshing

./allplot.sh png
