#!/usr/bin/env bash
# Lamb-Oseen vortex tutorial — generate all comparison figures.
#
# Calls one Python script per figure (all live in assets/).
# Output figures are written to figures/.
#
# Usage:
#   ./allplot.sh [--solution-dir PATH] [--dpi N]
#
# Defaults: --solution-dir ./solution  --dpi 400

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_plot() {
    local fmt
    for fmt in png pdf; do
        python "$@" --format "$fmt"
    done
}

# ── Parse optional overrides ─────────────────────────────────────────────────
SOLUTION_DIR="./solution"
DPI=300
THESIS_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        --thesis-dir)   THESIS_DIR="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p figures

FIGURES_DIR="$SCRIPT_DIR/figures"

echo "[allplot] solution-dir : $SOLUTION_DIR"
echo "[allplot] dpi          : $DPI"
echo ""

# ── Figure 1: Single vortex — radial profiles ─────────────────────────────
echo "[1/5] Lamb-Oseen radial profiles ..."
run_plot assets/plot_vortex_comparison.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --dt 0.04 \
    --total-time 20.0

# ── Figure 2: Vortex dipole — trajectory and core radius ─────────────────
echo "[2/5] Dipole comparison ..."
run_plot assets/plot_dipole_comparison.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI"

# ── Figure 3: Co-rotating merger — angle, core, separation ───────────────
echo "[3/5] Merging vortex comparison ..."
run_plot assets/plot_merging_comparison.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --a0-over-b0 0.125 \
    --total-time 20.0

# ── Figure 4: z = 0 surface fields — tiled quadrant comparison ───────────
echo "[4/5] z=0 surface fields ..."
run_plot assets/plot_vortex_surface_fields.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI"

# ── Figure 5: Energy balance — dE/dt vs -nuΩ ──────────────────────────────
echo "[5/5] Energy balance ..."
run_plot assets/plot_lamboseen_energy.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI"

echo ""
echo "[allplot] Done. Figures saved to $FIGURES_DIR"
