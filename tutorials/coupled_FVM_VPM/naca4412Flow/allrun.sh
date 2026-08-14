#!/usr/bin/env bash
# Run the coupled NACA 4412 case and check it.
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
python -u naca4412_setup.py 2>&1 | tee solution/naca4412.log

echo
echo "===== VALIDATE ====="
echo
python assets/check_run.py

echo
echo "===== DONE ====="
echo
echo "Simulation completed. Run ./allplot.sh to make the figures."