#!/usr/bin/env bash
# Make every quadcopter diagnostic figure (particle count, vorticity history).
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

plot() {
    python "$@" --format "$format"
}

plot assets/plot_quadcopter_particle_count.py
plot assets/plot_quadcopter_vorticity_history.py

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"