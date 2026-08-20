#!/usr/bin/env bash
# Run the whole step-profile tutorial: clean, simulate, make the figures.
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
# Laminar backward-facing-step flow at Re_h=100. The mesh contains the solid
# upstream block and the vertical step; this is not a scalar step-profile case.
python step_profile_setup.py --Re 100 --end-time 12 2>&1 | tee solution/step_profile.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo