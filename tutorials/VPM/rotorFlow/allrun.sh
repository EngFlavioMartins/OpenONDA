#!/usr/bin/env bash
# RotorFlow — VLM-VPM rotor-wake certification (3500 steps).
# Convects the wake through 3D, 6D, and 9D validation planes.
# Output logged to /tmp/rotor_run.log (safe from allclean.sh).
set -euo pipefail
PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

LOG=/tmp/rotor_run.log
echo "Logging to $LOG"

"$PYTHON" rotor_setup.py --num-steps 3500 > "$LOG" 2>&1

./allplot.sh
echo "All runs and plots complete."
