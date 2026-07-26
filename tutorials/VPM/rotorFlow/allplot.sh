#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution/rotor"
DPI=300

run_plot() {
    local fmt
    for fmt in png pdf; do
        "$PYTHON" "$@" --format "$fmt"
    done
}

FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi) DPI="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Plotting a live run bakes a startup transient into figures/ that is
# indistinguishable from physics: the downstream sampler planes read close to
# the freestream simply because the wake front has not reached them yet.
if [[ "$FORCE" -eq 0 ]] && pgrep -f "rotor_setup\.py" > /dev/null 2>&1; then
    echo "ERROR: a rotorFlow simulation is still running." >&2
    echo "       Figures made now would show a wake that is still in transit," >&2
    echo "       not a converged one.  Wait for the run to finish, or pass" >&2
    echo "       --force if you deliberately want a mid-run snapshot." >&2
    exit 1
fi

mkdir -p figures

echo "[1/3] Rotor performance"
run_plot assets/plot_rotor_performance.py \
    --solution-dir "$SOLUTION_DIR" --dpi "$DPI"

echo "[2/3] Rotor wake planes"
run_plot assets/plot_rotor_wake_planes.py \
    --solution-dir "$SOLUTION_DIR" --dpi "$DPI"

echo "[3/3] Spanwise loading vs BEM"
run_plot assets/plot_rotor_loading_validation.py \
    --solution-dir "$SOLUTION_DIR" --dpi "$DPI"

echo "Figures written to: $SCRIPT_DIR/figures"
