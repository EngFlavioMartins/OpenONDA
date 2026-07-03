#!/usr/bin/env bash
# RotorFlow — VLM-VPM rotor-wake tutorial.
# Convects the wake through 3D, 6D, and 9D validation planes.
set -euo pipefail
PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./allclean.sh

"$PYTHON" rotor_setup.py --num-steps "${N_STEPS:-2400}" --dt "${DT:-0.006}" --processing-unit CUDA

./allplot.sh
