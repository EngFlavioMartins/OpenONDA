#!/usr/bin/env bash
# Make every rotor figure (performance, wake planes, loading validation).
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

plot assets/plot_rotor_performance.py
plot assets/plot_rotor_wake_planes.py
plot assets/plot_rotor_loading_validation.py

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"