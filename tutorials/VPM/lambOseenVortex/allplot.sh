#!/usr/bin/env bash
# Generate every Lamb--Oseen comparison figure from samples/<case>/.

set -euo pipefail

cd "$(dirname "$0")"

case "${1:--png}" in
    -png) figure_format=png ;;
    -pdf) figure_format=pdf ;;
    *)
        echo "Usage: $0 [-png|-pdf]" >&2
        exit 2
        ;;
esac

if (($# > 1)); then
    echo "Usage: $0 [-png|-pdf]" >&2
    exit 2
fi

mkdir -p figures
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/openonda-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

plot() {
    if ! python "$@" --format "$figure_format"; then
        plot_status=1
    fi
}

plot_status=0
plot assets/plot_vortex_comparison.py
plot assets/plot_dipole_comparison.py
plot assets/plot_merging_comparison.py
plot assets/plot_vortex_surface_fields.py
plot assets/plot_lamboseen_energy.py

if ((plot_status)); then
    echo "One or more figures failed to render." >&2
    exit 1
fi

echo "Available $figure_format publication figures refreshed in figures/."
