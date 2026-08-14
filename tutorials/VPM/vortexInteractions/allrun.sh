#!/usr/bin/env bash
# Run every vortex-ring interaction case (or the subset given on the command
# line), check the results, and make the figures.
# Usage: ./allrun.sh [CASE ...]
set -euo pipefail

cd "$(dirname "$0")"

CASES="leapfrog_dns leapfrog_les leapfrog_les_stabilized collide_dns collide_les collide_les_stabilized"

echo
echo "===== CLEAN ====="
echo
./allclean.sh --all

echo
echo "===== SIMULATE ====="
echo
for case_name in ${*:-$CASES}; do
    echo
    echo "---- $case_name ----"
    python -u rings_setup.py --case "$case_name"
done

echo
echo "===== VALIDATE ====="
echo
python assets/check_run.py "$@"

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo