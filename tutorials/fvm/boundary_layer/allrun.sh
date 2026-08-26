#!/usr/bin/env bash
# Run the whole boundary-layer tutorial: clean, simulate, make the figures.
# Usage: ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "===== CLEAN ====="
echo
./allclean.sh --all

echo
echo "===== SIMULATE ====="
echo
mkdir -p solution
python boundary_layer_setup.py --Re 1e4 --end-time 8 2>&1 | tee solution/boundary_layer.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo