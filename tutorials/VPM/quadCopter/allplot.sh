#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOLUTION_DIR="./solution/quadcopter"

run_plot() {
    local fmt
    for fmt in png pdf; do
        python "$@" --format "$fmt"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p figures

echo "[1/2] Quadcopter particle count"
run_plot assets/plot_quadcopter_particle_count.py --solution-dir "$SOLUTION_DIR"

echo "[2/2] Quadcopter vorticity history"
run_plot assets/plot_quadcopter_vorticity_history.py --solution-dir "$SOLUTION_DIR"

echo "Figures written to: $SCRIPT_DIR/figures"
