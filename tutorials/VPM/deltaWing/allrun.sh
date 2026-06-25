#!/usr/bin/env bash
# Delta wing — VLM-VPM certification at high angle of attack.
# Captures leading-edge vortex convection and spanwise circulation.
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

"$PYTHON" delta_wing_setup.py

./allplot.sh
echo "All runs and plots complete."
