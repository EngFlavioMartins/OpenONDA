#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SOLUTION_DIR="./solution"; DPI=300

run_plot() {
    local fmt
    for fmt in png pdf; do
        python "$@" --format "$fmt"
    done
}

while [[ $# -gt 0 ]]; do case "$1" in
    --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
    --dpi) DPI="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac; done
mkdir -p figures
FIGURES_DIR="$SCRIPT_DIR/figures"

run_plot assets/plot_plate_polar.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_plate_staticvsmoving.py --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI"

run_plot assets/plot_plate_spanwise.py \
    --case-moving exp_moving_aoa05 \
    --case-static exp_static_aoa05 \
    --aoa 5 \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir "$FIGURES_DIR" --dpi "$DPI"

# Kelvin's theorem (bound vs shed circulation) for the static AoA=8 case.
run_plot assets/plot_flat_plate_kelvin.py --name exp_static_aoa08 \
    --solution-dir "$SOLUTION_DIR" --figures-dir "$FIGURES_DIR" --dpi "$DPI" || \
    echo "  (Kelvin plot skipped — run exp_static_aoa08 with --sample-crossflow/backups first)"

echo "Done. Figures saved to $FIGURES_DIR"
