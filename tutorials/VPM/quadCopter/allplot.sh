#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution/quadcopter"
FORMAT="png"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p figures

echo "[1/2] Quadcopter particle count"
"$PYTHON" assets/plot_quadcopter_particle_count.py --solution-dir "$SOLUTION_DIR" --format "$FORMAT"

echo "[2/2] Quadcopter vorticity history"
"$PYTHON" assets/plot_quadcopter_vorticity_history.py --solution-dir "$SOLUTION_DIR" --format "$FORMAT"

echo "Figures written to: $SCRIPT_DIR/figures"
