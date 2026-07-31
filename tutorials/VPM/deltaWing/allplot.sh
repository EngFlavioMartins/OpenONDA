#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOLUTION_DIR="./solution/delta_wing"

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

echo "[1/2] Delta-wing forces"
run_plot assets/plot_delta_wing_forces.py --solution-dir "$SOLUTION_DIR"

echo "[2/2] Delta-wing circulation history"
run_plot assets/plot_delta_wing_circulation_history.py --solution-dir "$SOLUTION_DIR"

echo "Figures written to: $SCRIPT_DIR/figures"
