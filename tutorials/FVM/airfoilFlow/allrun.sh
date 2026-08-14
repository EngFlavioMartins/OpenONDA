#!/usr/bin/env bash
# Run the whole airfoil tutorial: clean, simulate, make the figures.
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
python airfoilFlow_setup.py --Re 1000 --angle 0 --end-time 25 2>&1 | tee solution/airfoilFlow.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo