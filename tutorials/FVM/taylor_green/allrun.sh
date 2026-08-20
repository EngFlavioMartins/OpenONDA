#!/usr/bin/env bash
# Run the whole Taylor-Green tutorial: clean, simulate, make the figures.
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
python taylor_green_setup.py --n 24 --nu 0.1 --dt 0.005 --end-time 0.05 \
    2>&1 | tee solution/taylor_green.log

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo