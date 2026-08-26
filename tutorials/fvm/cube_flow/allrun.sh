#!/usr/bin/env bash
# Run the whole cube-flow tutorial: clean, simulate, make the figures.
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
# Square-cylinder von Karman street at Re = 100 (mesh built in-memory).
# Validation: St = 0.140-0.150, mean Cd = 1.45-1.58
# (Okajima 1982; Sohankar et al. 1998; Sen et al. 2011).
python cube_flow_setup.py --Re 100 --end-time 120 2>&1 | tee solution/cube_flow.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo