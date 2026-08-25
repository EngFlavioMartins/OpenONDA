#!/usr/bin/env bash
# Make every vortex-ring interaction figure (six-case diagnostics).
#
# Usage:
#   ./allplot.sh        PNG figures (default)
#   ./allplot.sh pdf    PDF figures
set -euo pipefail

cd "$(dirname "$0")"

format="${1:-png}"
allow_partial="${2:-}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;;
esac
if [[ -n "$allow_partial" && "$allow_partial" != "--allow-partial" ]]; then
    echo "Usage: $0 [png|pdf] [--allow-partial]" >&2
    exit 2
fi

figures_dir="${VPM_INTERACTIONS_FIGURES_DIR:-figures}"
mkdir -p "$figures_dir"

echo
echo "===== FIGURES ($format) ====="
echo

if [[ "$allow_partial" == "--allow-partial" ]]; then
    python assets/validate_plot_inputs.py --allow-partial
else
    python assets/validate_plot_inputs.py
fi

plot() {
    python "$@" --format "$format" --figures-dir "$figures_dir"
}

plot assets/plot_rings_circulation.py
plot assets/plot_rings_energy_budget.py
plot assets/plot_rings_energy.py
plot assets/plot_rings_conservation.py
plot assets/plot_rings_resolution.py
plot assets/plot_rings_stability.py
plot assets/plot_rings_trajectory.py

echo
echo "===== DONE ====="
echo "Figures saved to: $figures_dir/"
