#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark (single vortex, dipole, merging
# pair, for every diffusion scheme), then make the figures.
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
echo "---- single vortex (all schemes) ----"
python -u lambossen_setup.py --gamma1 +1
echo
echo "---- vortex dipole (all schemes) ----"
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1
echo
echo "---- merging pair (all schemes) ----"
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo