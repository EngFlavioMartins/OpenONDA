#!/usr/bin/env bash
# Delta wing — VLM-VPM certification at high angle of attack.
# Captures leading-edge vortex convection and spanwise circulation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

python delta_wing_setup.py --num-steps "${N_STEPS:-2200}" --dt "${DT:-0.004}" --processing-unit CUDA

./allplot.sh
echo "All runs and plots complete."
