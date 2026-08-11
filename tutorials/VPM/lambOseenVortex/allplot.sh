#!/usr/bin/env bash
# Generate every Lamb--Oseen comparison figure from samples/<case>/.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p figures
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/openonda-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

plot() {
    for format in png pdf; do
        if ! python "$@" --format "$format"; then
            plot_status=1
        fi
    done
}

plot_status=0
plot assets/plot_vortex_comparison.py
plot assets/plot_dipole_comparison.py
plot assets/plot_merging_comparison.py
plot assets/plot_vortex_surface_fields.py
plot assets/plot_lamboseen_energy.py

if ((plot_status)); then
    echo "Figures are incomplete; finish the missing solver cases and run allplot.sh again." >&2
    exit 1
fi

echo "All publication figures written to figures/."
