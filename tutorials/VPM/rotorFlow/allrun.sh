#!/usr/bin/env bash
# Run the rotor-wake case, check the results, and make the figures.
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
python rotor_setup.py 2>&1 | tee solution/rotor.log

echo
echo "===== VALIDATE ====="
echo
python assets/validate_results.py

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo