#!/usr/bin/env bash
# Single vortex ring — DNS/LES simulation and plotting script.
#
# Runs four configurations:
# 1. DNS  (no model,            direct/classical stretching)
# 2. DNS  (no model,            transposed stretching)
# 3. DNS  (no model,            mixed stretching)
# 4. LES  (Smagorinsky + CS,    transposed stretching, Pedrizzetti ISR α=0.95)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$(conda run -n OpenONDA which python 2>/dev/null \
       || command -v python3 \
       || command -v python)"

# Clean previous solution
./allclean.sh

echo "Starting DNS simulation (direct stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching direct --name DNS_direct --solution-dir ./solution

echo "Starting DNS simulation (transposed stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching transposed --name DNS_transposed --solution-dir ./solution

echo "Starting DNS simulation (mixed stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching mixed --name DNS_mixed --solution-dir ./solution

echo "Starting LES simulation (transposed stretching, Pedrizzetti ISR α=0.95)..."
"$PYTHON" ring_setup.py --mode les --stretching transposed --name LES_transposed \
    --direction-alpha 0.95 --solution-dir ./solution

echo "Generating comparison plots..."
./allplot.sh

echo
echo "All runs and plots complete."
