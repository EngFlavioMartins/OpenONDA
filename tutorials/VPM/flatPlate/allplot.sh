#!/usr/bin/env bash
PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null || command -v python3 || command -v python)"
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

python assets/plot_plate_polar.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

python assets/plot_plate_staticvsmoving.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

python assets/plot_plate_spanwise.py \
    --case-moving exp_moving_aoa05 \
    --case-static exp_static_aoa05 \
    --aoa 5 \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" --dpi "$DPI"

echo "Done. Figures saved to $FIGURES_DIR"
