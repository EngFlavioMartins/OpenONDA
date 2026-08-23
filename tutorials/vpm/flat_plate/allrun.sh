#!/usr/bin/env bash
# Run the moving- and fixed-plate angle-of-attack sweeps, then make the figures.
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
for angle in -10 -5 -2 0 2 5 8 10 12 15; do
    echo
    echo "---- moving plate, angle $angle deg ----"
    python setup_plate.py --mode moving --angle "$angle"
done

for angle in -10 -5 -2 0 2 5 8 10 12 15; do
    echo
    echo "---- static plate, angle $angle deg ----"
    python setup_plate.py --mode static --angle "$angle"
done

echo
echo "===== FIGURES ====="
echo
./plot_all.sh

echo
echo "===== DONE ====="
echo