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

# Resolve Python interpreter: same precedence as allrun.sh.
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null \
       || command -v python3 \
       || command -v python)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse optional overrides ─────────────────────────────────────────────────
SOLUTION_DIR="./solution"
DPI=300
FORMAT="png"
THESIS_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        --format)       FORMAT="$2";       shift 2 ;;
        --thesis-dir)   THESIS_DIR="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$FORMAT" != "png" && "$FORMAT" != "svg" ]]; then
    echo "Unknown format: $FORMAT (must be png or svg)" >&2; exit 1
fi

mkdir -p figures

FIGURES_DIR="$SCRIPT_DIR/figures"

echo "[allplot] solution-dir : $SOLUTION_DIR"
echo "[allplot] dpi          : $DPI"
echo "[allplot] format       : $FORMAT"
echo ""

# ── Figure 1: Single vortex — radial profiles ─────────────────────────────
echo "[1/5] Lamb-Oseen radial profiles ..."
"$PYTHON" assets/plot_vortex_comparison.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --format "$FORMAT"

# ── Figure 2: Vortex dipole — trajectory and core radius ─────────────────
echo "[2/5] Dipole comparison ..."
"$PYTHON" assets/plot_dipole_comparison.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --format "$FORMAT"

# ── Figure 3: Co-rotating merger — angle, core, separation ───────────────
echo "[3/5] Merging vortex comparison ..."
"$PYTHON" assets/plot_merging_comparison.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --format "$FORMAT"

# ── Figure 4: z = 0 surface fields — tiled quadrant comparison ───────────
echo "[4/5] z=0 surface fields ..."
"$PYTHON" assets/plot_vortex_surface_fields.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --format "$FORMAT"

# ── Figure 5: Energy balance — dE/dt vs -nuΩ ──────────────────────────────
echo "[5/5] Energy balance ..."
"$PYTHON" assets/plot_energy_enstrophy.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" \
    --format "$FORMAT"

echo ""
echo "[allplot] Done. Figures saved to $FIGURES_DIR"
