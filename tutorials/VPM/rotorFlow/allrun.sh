#!/usr/bin/env bash
# RotorFlow VLM-VPM — full certification run
#
# Runs 8 rotor revolutions (1232 steps at dt=0.005 s, TSR=7, R=6 m).
# Eight revolutions ensure:
#   - wake develops past 4R downstream (required for induction figures)
#   - Ct/Cp approach a stationary average over the last revolution
#
# Output redirect: /tmp/rotor_run.log  (outside case dir, safe from allclean.sh)
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

LOG=/tmp/rotor_run.log
echo "Logging to $LOG"

"$PYTHON" rotor_setup.py --num-steps 1232 > "$LOG" 2>&1

./allplot.sh
echo "All runs and plots complete."
