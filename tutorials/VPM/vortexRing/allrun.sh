#!/usr/bin/env bash
# Vortex ring — stretching-formulation benchmark (direct, transposed, mixed)
# and LES comparison (rVPM correction vs plain transposed stretching).
# Runs five configurations; strength relaxation is intentionally not part of
# the certified Saffman benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
       || command -v python3 \
       || command -v python)}"

# Clean previous solution
./allclean.sh

COMMON_FLAGS=(--solution-dir ./solution --processing-unit CUDA)

echo "Starting DNS simulation (direct stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching direct --name DNS_direct "${COMMON_FLAGS[@]}"

echo "Starting DNS simulation (transposed stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching transposed --name DNS_transposed "${COMMON_FLAGS[@]}"

echo "Starting DNS simulation (mixed stretching)..."
"$PYTHON" ring_setup.py --mode dns --stretching mixed --name DNS_mixed "${COMMON_FLAGS[@]}"

echo "Starting LES simulation (transposed stretching + rVPM correction)..."
"$PYTHON" ring_setup.py --mode les --stretching transposed --parallel-strain-relaxation --name LES_rvpm "${COMMON_FLAGS[@]}"

echo "Starting LES comparison simulation (transposed stretching)..."
"$PYTHON" ring_setup.py --mode les --stretching transposed --name LES_transposed "${COMMON_FLAGS[@]}"


echo "Generating comparison plots..."
./allplot.sh

echo
echo "All runs and plots complete."
