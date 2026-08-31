#!/usr/bin/env bash
# Make every cylinder-shedding figure (lift history, probe growth, growth fit,
# onset alignment, spectrum, spanwise coherence, midspan vorticity).
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

export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
mkdir -p figures

echo
echo "===== FIGURES ($format) ====="
echo

python assets/plot_reference_mesh.py --format "$format"
if [ -f reference_flow/solution/grid_independence.json ]; then
    python assets/plot_grid_independence.py --format "$format"
else
    echo "Skipping grid_independence: final convergence report is not available yet."
fi
python assets/plot_von_karman.py --format "$format"

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"
