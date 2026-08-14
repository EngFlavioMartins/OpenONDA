#!/usr/bin/env bash
# Run the delta-wing wake-crossing case, then make the figures.
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
python delta_wing_setup.py 2>&1 | tee solution/delta_wing.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo