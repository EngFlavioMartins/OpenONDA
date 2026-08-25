#!/usr/bin/env bash
# Make every vortex-ring figure (motion, energy, circulation).
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

plot assets/plot_vortex_ring_motion.py
plot assets/plot_vortex_ring_energy.py
plot assets/plot_vortex_ring_circulation.py

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"