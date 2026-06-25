#!/usr/bin/env bash
# Vortex ring — stretching-formulation benchmark (direct, transposed, mixed)
# and LES comparison (Pedrizzetti ISR vs rVPM + blend ISR).
# Runs five configurations with Smagorinsky LES on the transposed cases.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
       || command -v python3 \
       || command -v python)}"

# Clean previous solution
./allclean.sh

echo "Starting DNS simulation (direct stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching direct --name DNS_direct --solution-dir ./solution

echo "Starting DNS simulation (transposed stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching transposed --name DNS_transposed --solution-dir ./solution

echo "Starting DNS simulation (mixed stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching mixed --name DNS_mixed --solution-dir ./solution

echo "Starting LES comparison simulation (transposed stretching, Pedrizzetti ISR α=0.95)..."
"$PYTHON" ring_setup.py --mode les --stretching transposed --name LES_transposed \
    --relaxation pedrizzetti --relaxation-factor 0.95 --solution-dir ./solution

echo "Starting LES simulation (rVPM reformulated stretching, Alvarez & Ning)..."
"$PYTHON" ring_setup.py --mode les --stretching rvpm --name LES_rvpm \
    --rvpm-g 0.3333333333333333 --relaxation blend --solution-dir ./solution

echo "Generating comparison plots..."
./allplot.sh

echo
echo "All runs and plots complete."
