#!/usr/bin/env bash
# Make every cube-flow figure (forces, vorticity, wake) from the results.
#
# Usage:
#   ./plot_all.sh        PNG figures (default)
#   ./plot_all.sh pdf    PDF figures
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

plot assets/plot_forces.py
plot assets/plot_vorticity.py
plot assets/plot_wake.py

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"