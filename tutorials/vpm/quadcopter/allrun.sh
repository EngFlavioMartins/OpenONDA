#!/usr/bin/env bash
# Run the quadcopter climb case, then make the figures.
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
python quadcopter_setup.py 2>&1 | tee solution/quadcopter.log

echo
echo "===== FIGURES ====="
echo
python assets/validate_results.py --pre-plot
./allplot.sh png
./allplot.sh pdf

echo
echo "===== VALIDATE ====="
echo
python assets/validate_results.py

echo
echo "===== DONE ====="
echo
