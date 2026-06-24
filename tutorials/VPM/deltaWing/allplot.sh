#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution/delta_wing"

if [[ ${1:-} == "--solution-dir" ]]; then
    SOLUTION_DIR="$2"
fi

mkdir -p figures

echo "[1/2] Delta-wing forces"
"$PYTHON" assets/plot_delta_wing_forces.py --solution-dir "$SOLUTION_DIR"

echo "[2/2] Delta-wing circulation history"
"$PYTHON" assets/plot_delta_wing_circulation_history.py --solution-dir "$SOLUTION_DIR"

echo "Figures written to: $SCRIPT_DIR/figures"
