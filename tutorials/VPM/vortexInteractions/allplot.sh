#!/usr/bin/env bash
# Make every vortex-ring interaction figure (six-case diagnostics).
#
# Usage:
#   ./allplot.sh        PNG figures (default)
#   ./allplot.sh pdf    PDF figures
set -euo pipefail

cd "$(dirname "$0")"

format="${1:-png}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;;
esac

mkdir -p figures

echo
echo "===== FIGURES ($format) ====="
echo

python assets/validate_plot_inputs.py

plot() {
    python "$@" --format "$format"
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
echo "Figures saved to: figures/"