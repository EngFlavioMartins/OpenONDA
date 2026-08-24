#!/usr/bin/env bash
# Run every vortex-ring physics case, then make the comparison figures.
# Usage: ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"
export OPENONDA_COMPUTE_DEVICE="${OPENONDA_COMPUTE_DEVICE:-METAL}"

echo
echo "===== CLEAN ====="
echo
./allclean.sh
mkdir -p solution

echo
echo "===== SIMULATE ====="
echo
for variant in dns_direct dns_transposed dns_mixed les_transposed; do
    echo
    echo "---- vortex ring: $variant ----"
    python -u ring_setup.py --variant "$variant" \
        --compute-device "$OPENONDA_COMPUTE_DEVICE" \
        --velocity-method TREECODE --treecode-theta 0.30 \
        2>&1 | tee "solution/${variant}.log"
done

echo
echo "===== FIGURES ====="
echo
python assets/validate_results.py --pre-plot
./plot_all.sh png
./plot_all.sh pdf

echo
echo "===== VALIDATE ====="
echo
python assets/validate_results.py

echo
echo "===== DONE ====="
echo
