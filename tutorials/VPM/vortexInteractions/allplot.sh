#!/usr/bin/env bash
# Vortex-ring interactions — comparison figures across the stabilization ladder.
#
# Each script auto-discovers the intended six-case matrix under the solution
# directory: leapfrog_{dns,les,les_isr} and collide_{dns,les,les_isr}.  The
# figures share one styling key: colour = stabilization rung, linestyle =
# physics family (leapfrog solid, collide dashed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"
SOLUTION_DIR="./solution"
FIGURES_DIR="./figures"
DPI=300

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --figures-dir) FIGURES_DIR="$2"; shift 2 ;;
        --dpi) DPI="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$FIGURES_DIR"

echo "[1/5] Ring trajectory vs LBM reference  ->  compare_trajectory.png"
"$PYTHON" assets/plot_compare_trajectory.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "[2/5] Total circulation conservation    ->  rings_circulation.png"
"$PYTHON" assets/plot_rings_circulation.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "[3/5] Energy and enstrophy evolution     ->  rings_energy.png"
"$PYTHON" assets/plot_rings_energy.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "[4/5] Stability ladder (max|Γ| blow-up)  ->  rings_stability.png"
"$PYTHON" assets/plot_rings_stability.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "[5/5] Energy-budget identity audit       ->  energy_budget_audit.png"
"$PYTHON" assets/analyze_energy_budget.py \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "Figures written to: $FIGURES_DIR"
