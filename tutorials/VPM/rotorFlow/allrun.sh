#!/usr/bin/env bash
# RotorFlow VLM-VPM — full certification run
#
# Runs 3500 steps (17.5 s at dt=0.005 s), long enough for the wake to
# convect through the 3D/6D/9D validation planes.
#
# Output redirect: /tmp/rotor_run.log  (outside case dir, safe from allclean.sh)
set -euo pipefail
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

LOG=/tmp/rotor_run.log
echo "Logging to $LOG"

"$PYTHON" rotor_setup.py --num-steps 3500 > "$LOG" 2>&1

./allplot.sh
echo "All runs and plots complete."
