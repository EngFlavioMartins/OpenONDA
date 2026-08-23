#!/usr/bin/env bash
# Run every vortex-ring physics case, then make the comparison figures.
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
for variant in dns_direct dns_transposed dns_mixed les_transposed; do
    echo
    echo "---- vortex ring: $variant ----"
    python ring_setup.py --variant "$variant"
done

echo
echo "===== FIGURES ====="
echo
./plot_all.sh

echo
echo "===== DONE ====="
echo