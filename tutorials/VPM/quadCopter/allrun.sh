#!/usr/bin/env bash
# Quadcopter — VLM-VPM rotor-rotor interaction with wake advection.
# Demonstrates multi-rotor VPM coupling and periodic hover convergence.
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

"$PYTHON" quad_setup.py

./allplot.sh
echo "All runs and plots complete."
