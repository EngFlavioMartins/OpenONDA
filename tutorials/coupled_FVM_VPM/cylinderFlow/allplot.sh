#!/usr/bin/env bash
# Plot all matched cylinder diagnostics from sampler output only.
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR" figures

format="${1:-png}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;;
esac

python assets/plot_forces.py --format "$format"
python assets/plot_velocity_profiles.py --format "$format"
python assets/plot_velocity_fields.py --format "$format"
python assets/plot_coupling_diagnostics.py --format "$format"
python assets/compare_results.py

echo "Wrote cylinder figures to figures/ and metrics to samples/comparison_metrics.json."
