#!/usr/bin/env bash
# RotorFlow — VLM-VPM rotor-wake tutorial.
# Convects the wake through 3D, 6D, and 9D validation planes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SOLUTION_DIR="./solution/rotor"

./allclean.sh

# PROCESSING_UNIT defaults to AUTO, which selects Metal on macOS and
# CUDA/Vulkan elsewhere.  Override (e.g. PROCESSING_UNIT=CUDA) to pin a backend.
python rotor_setup.py --num-steps "${N_STEPS:-2400}" --dt "${DT:-0.006}" \
    --processing-unit "${PROCESSING_UNIT:-AUTO}"

./allplot.sh

# Acceptance checks: wake-plane convergence, BEM agreement, and the wake's
# impulse budget.  Runs last so a certification failure is the script's exit
# code without suppressing the figures.
python assets/validate_results.py \
    --solution-dir "$SOLUTION_DIR" --expected-step "${N_STEPS:-2400}"
