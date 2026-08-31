#!/usr/bin/env bash
# Run the three leapfrogging LES stabilization comparisons.
set -euo pipefail

cd "$(dirname "$0")"

rm -rf solution samples figures
mkdir -p solution samples figures

python -u interactions_setup.py --case leapfrog_les
python -u interactions_setup.py --case leapfrog_les_splitting
python -u interactions_setup.py --case leapfrog_les_splitting_remeshing

./allplot.sh png
