#!/usr/bin/env bash
# Make every Lamb--Oseen comparison figure from samples/<case>/.
#
# Usage:
#   ./allplot.sh        PNG figures (default)
#   ./allplot.sh pdf    PDF figures
set -euo pipefail

cd "$(dirname "$0")"

# Keep headless plotting deterministic on machines whose home cache is not
# writable (common in CI/sandboxed runs) and avoid rebuilding the font cache
# once for every figure script.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/openonda-matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

format="${1:-png}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;;
esac

mkdir -p figures

echo
echo "===== FIELD DIAGNOSTICS ====="
echo
python assets/vortex_diagnostics.py

echo
echo "===== FIGURES ($format) ====="
echo

plot() {
    python "$@" --format "$format"
}

plot assets/plot_vortex_comparison.py
plot assets/plot_dipole_comparison.py
plot assets/plot_merging_comparison.py
plot assets/plot_vortex_surface_fields.py
plot assets/plot_lamboseen_energy.py

echo
echo "===== DATA STATUS ====="
echo
python assets/postprocess.py --manifest

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"
