#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# NACA 0012 at Re = 1000 and zero angle of attack.
# The force and pressure plots report the expected symmetry diagnostics.
python airfoilFlow_setup.py --Re 1000 --angle 0 --end-time 25 --dt 0.005 \
    2>&1 | tee solution/airfoilFlow.log

./allplot.sh --angle 0
echo
echo "All runs and plots complete."
