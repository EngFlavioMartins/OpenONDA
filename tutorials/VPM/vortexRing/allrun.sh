#!/usr/bin/env bash
# Single vortex ring — DNS/LES simulation and plotting script.
#
# Runs four configurations:
# 1. DNS simulation (no model)
# 2. LES simulation (Smagorinsky, CS viscous, direct/classical stretching)
# 3. LES simulation (Smagorinsky, CS viscous, transposed stretching)
# 4. LES simulation (Smagorinsky, CS viscous, mixed stretching)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clean previous solution
./allclean.sh

echo "Starting DNS simulation (direct stretching)..."
python ring_setup.py --mode dns --stretching direct --name DNS_direct --solution-dir ./solution

echo "Starting DNS simulation (transposed stretching)..."
python ring_setup.py --mode dns --stretching transposed --name DNS_transposed --solution-dir ./solution

echo "Starting DNS simulation (mixed stretching)..."
python ring_setup.py --mode dns --stretching mixed --name DNS_mixed --solution-dir ./solution

python ring_setup.py --mode les --stretching transposed --name LES_transposed --direction-alpha 0.95

echo "Generating comparison plots..."
./allplot.sh

echo
echo "All runs and plots complete."
