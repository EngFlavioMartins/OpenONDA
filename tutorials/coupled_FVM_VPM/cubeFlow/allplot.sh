#!/usr/bin/env bash
# Make every coupled cube-flow figure (velocity profiles, fields, wake errors,
# forces) from the sampled results.
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

plot assets/plot_velocity_profiles.py
plot assets/plot_velocity_fields.py
plot assets/plot_wake_errors.py
plot assets/plot_forces.py

echo
echo "===== DONE ====="
echo "Figures saved to: figures/"