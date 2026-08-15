#!/usr/bin/env bash
# Make every coupled cube-flow figure (velocity profiles, fields, wake errors,
# forces) from the sampled results.
#
# Usage:
#   ./allplot.sh        PNG figures (default)
#   ./allplot.sh pdf    PDF figures
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

format="${1:-png}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;;
esac

mkdir -p figures

plot() {
    python "$@" --format "$format"
}

plot assets/plot_velocity_profiles.py
plot assets/plot_velocity_fields.py
plot assets/plot_coupling_diagnostics.py
# plot assets/plot_wake_errors.py
