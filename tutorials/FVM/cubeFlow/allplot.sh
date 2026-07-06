#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
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
echo ""
echo "[1/3] Forces ..."
python assets/plot_forces.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI" --format "$FORMAT"
echo "[2/3] Wake centreline ..."
python assets/plot_wake.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI" --format "$FORMAT"
echo "[3/3] Vorticity slice ..."
python assets/plot_vorticity.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI" --format "$FORMAT"
echo ""
echo "[allplot] Done. Figures saved to $FIGURES_DIR"
