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
for variant in DNS_direct DNS_transposed DNS_mixed LES_transposed; do
    echo
    echo "---- vortex ring: $variant ----"
    python ring_setup.py --variant "$variant"
done

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo