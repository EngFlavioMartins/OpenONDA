#!/usr/bin/env bash
# Run the delta-wing wake-crossing case, then make the figures.
# Usage: ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"
export OPENONDA_COMPUTE_DEVICE="${OPENONDA_COMPUTE_DEVICE:-METAL}"

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
mkdir -p solution
python delta_wing_setup.py > solution/delta_wing.log 2>&1

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
