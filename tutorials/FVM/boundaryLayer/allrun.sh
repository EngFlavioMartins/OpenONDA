#!/usr/bin/env bash
# Run the whole boundary-layer tutorial: clean, simulate, make the figures.
# Usage: ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
mkdir -p solution
python boundaryLayer_setup.py --Re 1e4 --end-time 8 2>&1 | tee solution/boundaryLayer.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo