#!/usr/bin/env bash
# Vortex-ring interactions — baseline, LES, and stabilized-LES comparison.
#
# Each script auto-discovers the three methods for both interaction families.
# Color identifies the numerical method and linestyle identifies the family.
set -euo pipefail

PLOT_CACHE_ROOT="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/openonda-plot-cache}"
export XDG_CACHE_HOME="$PLOT_CACHE_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PLOT_CACHE_ROOT/matplotlib}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution"
FIGURES_DIR="./figures"
DPI=300
ALLOW_PARTIAL=0

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
        --allow-partial) ALLOW_PARTIAL=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

VALIDATE_ARGS=(--solution-dir "$SOLUTION_DIR")
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
    VALIDATE_ARGS+=(--allow-partial)
fi
"$PYTHON" assets/validate_plot_inputs.py "${VALIDATE_ARGS[@]}"

mkdir -p "$FIGURES_DIR"

run_plot assets/plot_rings_circulation.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_energy_budget.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_energy.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_conservation.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_stability.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_resolution.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_rings_trajectory.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "Figures written to: $FIGURES_DIR"
