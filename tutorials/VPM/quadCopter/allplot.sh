#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution/quadcopter"

if [[ ${1:-} == "--solution-dir" ]]; then
    SOLUTION_DIR="$2"
fi

mkdir -p figures

echo "[1/2] Quadcopter particle count"
"$PYTHON" assets/plot_quadcopter_particle_count.py --solution-dir "$SOLUTION_DIR"

echo "[2/2] Quadcopter vorticity history"
"$PYTHON" assets/plot_quadcopter_vorticity_history.py --solution-dir "$SOLUTION_DIR"

echo "Figures written to: $SCRIPT_DIR/figures"
