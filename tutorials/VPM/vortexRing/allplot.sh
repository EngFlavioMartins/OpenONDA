#!/usr/bin/env bash
# Single vortex ring — generate all comparison figures.
#
# Calls one Python script per figure (all live in assets/).
# Output figures are written to figures/.
#
# Usage:
#   ./allplot.sh [--solution-dir PATH] [--dpi N] [--format png|pdf]
#
# Defaults: --solution-dir ./solution  --dpi 400

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"

# ── Parse optional overrides ─────────────────────────────────────────────────
SOLUTION_DIR="./solution"
DPI=400
FORMAT="png"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        --format)       FORMAT="$2";       shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

FIGURES_DIR="$SCRIPT_DIR/figures"
mkdir -p "$FIGURES_DIR"

echo "[allplot] solution-dir : $SOLUTION_DIR"
echo "[allplot] figures-dir  : $FIGURES_DIR"
echo "[allplot] dpi          : $DPI"
echo "[allplot] format       : $FORMAT"
echo ""

# ── Figure 1: Ring self-induced velocity  U/U₀ vs normalized time ────────────
echo "[1/3] Ring self-induced velocity ..."
"$PYTHON" assets/plot_vortex_ring_motion.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR"  \
    --dpi "$DPI" \
    --format "$FORMAT"

# ── Figure 2: Energy dissipation  dE/dt & −nuΩ vs normalized time ────────────
echo "[2/3] Energy dissipation ..."
"$PYTHON" assets/plot_vortex_ring_energy.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR"  \
    --dpi "$DPI" \
    --format "$FORMAT"

# ── Figure 3: Tube circulation Γ/Γ₀ vs normalized time (all variants) ───────
echo "[3/3] Tube circulation (DNS + LES, all stretching variants) ..."
"$PYTHON" assets/plot_vortex_ring_circulation.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR"  \
    --dpi "$DPI" \
    --format "$FORMAT"

echo ""
echo "[allplot] Done. Figures saved to $FIGURES_DIR"
