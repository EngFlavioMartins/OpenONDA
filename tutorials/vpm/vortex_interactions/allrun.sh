#!/usr/bin/env bash
# Run every vortex-ring interaction case (or the subset given on the command
# line), check the results, and make the figures.
# Usage: ./allrun.sh [CASE ...]
set -euo pipefail

cd "$(dirname "$0")"
export OPENONDA_COMPUTE_DEVICE="${OPENONDA_COMPUTE_DEVICE:-METAL}"

CASES="leapfrog_dns leapfrog_les leapfrog_les_stabilized collide_dns collide_les collide_les_stabilized"

echo
echo "===== CLEAN ====="
echo
if [[ $# -eq 0 ]]; then
    ./allclean.sh --all
else
    for case_name in "$@"; do
        ./allclean.sh "$case_name"
    done
fi

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
if [[ $# -eq 0 ]]; then
    ./allplot.sh png
    ./allplot.sh pdf
else
    # Keep subset diagnostics separate from the canonical six-case figures;
    # this avoids silently overwriting a previously validated full campaign.
    VPM_INTERACTIONS_FIGURES_DIR="figures/partial" ./allplot.sh png --allow-partial
    VPM_INTERACTIONS_FIGURES_DIR="figures/partial" ./allplot.sh pdf --allow-partial
fi

echo
echo "===== DONE ====="
echo
