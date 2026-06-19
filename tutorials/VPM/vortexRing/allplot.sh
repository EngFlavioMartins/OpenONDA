#!/usr/bin/env bash
# Single vortex ring — generate all comparison figures.
#
# Calls one Python script per figure (all live in assets/).
# Output figures are written to figures/.
#
# Usage:
#   ./allplot.sh [--solution-dir PATH] [--dpi N]
#
# Defaults: --solution-dir ./solution  --dpi 400

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse optional overrides ─────────────────────────────────────────────────
SOLUTION_DIR="./solution"
DPI=400
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

FIGURES_DIR="$SCRIPT_DIR/figures"
mkdir -p "$FIGURES_DIR"

echo "[allplot] solution-dir : $SOLUTION_DIR"
echo "[allplot] figures-dir  : $FIGURES_DIR"
echo "[allplot] dpi          : $DPI"
echo ""

# ── Figure 1: Ring self-induced velocity  U/U₀ vs normalized time ────────────
echo "[1/3] Ring self-induced velocity ..."
python assets/plot_ring_motion.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR"  \
    --dpi "$DPI"

# ── Figure 2: Energy dissipation  dE/dt & −nuΩ vs normalized time ────────────
echo "[2/3] Energy dissipation ..."
python assets/plot_ring_energy.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR"  \
    --dpi "$DPI"

# ── Figure 3: Total circulation  Σ|Γᵢ|/Γ₀ vs normalized time (all variants) ──
echo "[3/3] Total circulation (DNS + LES, all stretching variants) ..."
python assets/plot_ring_circulation.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR"  \
    --dpi "$DPI"

echo ""
echo "[allplot] Done. Figures saved to $FIGURES_DIR"
