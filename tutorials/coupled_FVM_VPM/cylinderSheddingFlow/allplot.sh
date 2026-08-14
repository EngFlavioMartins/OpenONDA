#!/usr/bin/env bash
# Make the cylinder-shedding force figure from the sampled results.
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

python assets/plot_forces.py --format "$format"

echo
echo "===== DONE ====="
echo "Figure saved to: figures/"