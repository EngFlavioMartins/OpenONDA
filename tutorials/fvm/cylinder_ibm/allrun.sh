#!/usr/bin/env bash
# Run the whole cylinder-IBM tutorial: clean, simulate, make the figures.
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
# Steady validation case: Re = 30 cylinder (Constant et al. 2017).
# Quality targets: Cd = 1.74-1.80, recirculation L/D = 1.55-1.70.
# For the unsteady von Karman validation run instead:
#   Edit setup.py to configure the unsteady Re = 100 validation case.
#   ./allplot.sh
python setup.py 2>&1 | tee solution/cylinder_ibm.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo
