#!/usr/bin/env bash
PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SOLUTION_DIR="./solution"; DPI=300; FORMAT="png"
while [[ $# -gt 0 ]]; do case "$1" in
    --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
    --dpi) DPI="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac; done
mkdir -p figures
FIGURES_DIR="$SCRIPT_DIR/figures"

echo "[1/2] Rotor diagnostics ..."
"$PYTHON" assets/allplot.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" --format "$FORMAT"
echo "[2/2] Wake-plane validation vs momentum theory (3/6/9 D) ..."
"$PYTHON" assets/plot_wake_planes.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" \
    --dpi "$DPI" --format "$FORMAT"
echo "Done. Figures saved to $FIGURES_DIR"
