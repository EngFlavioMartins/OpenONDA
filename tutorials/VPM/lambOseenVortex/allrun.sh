#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark (single vortex, dipole, merging
# pair, for every diffusion scheme), then make the figures.
# Usage: ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"

SCHEMES=(cs rwm dvh gbd)

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
for scheme in "${SCHEMES[@]}"; do
    echo
    echo "---- single vortex ($scheme) ----"
    python -u lambossen_setup.py --gamma1 +1 --schemes "$scheme"
    echo
    echo "---- vortex dipole ($scheme) ----"
    python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --schemes "$scheme"
    echo
    echo "---- merging pair ($scheme) ----"
    python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --schemes "$scheme"
done

echo
echo "===== FIGURES ====="
echo
./allplot.sh

echo
echo "===== DONE ====="
echo