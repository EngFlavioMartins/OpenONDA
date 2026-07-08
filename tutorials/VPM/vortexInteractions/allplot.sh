#!/usr/bin/env bash
# Vortex-ring interactions — LES transposed stabilizer comparison.
#
# Each script auto-discovers the stabilizer matrix under the solution directory:
# leapfrog_* and collide_* with variants {les,rvpm,relax,remesh,projection,split}.
# The figures share one styling key: color = stabilization, linestyle = family.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution"
FIGURES_DIR="./figures"
DPI=300

run_plot() {
    local fmt
    for fmt in png pdf; do
        "$PYTHON" "$@" --format "$fmt"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --figures-dir) FIGURES_DIR="$2"; shift 2 ;;
        --dpi) DPI="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$FIGURES_DIR"

run_plot assets/plot_rings_circulation.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_energy_budget.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_energy.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_stability.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_trajectory.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "Figures written to: $FIGURES_DIR"
