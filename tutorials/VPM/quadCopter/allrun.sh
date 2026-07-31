#!/usr/bin/env bash
# Quadcopter — VLM-VPM rotor-rotor interaction with wake advection.
# Demonstrates multi-rotor VPM coupling and periodic hover convergence.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

python quad_setup.py --processing-unit CUDA

./allplot.sh
echo "All runs and plots complete."
