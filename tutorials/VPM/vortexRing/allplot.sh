#!/usr/bin/env bash
# Generate the vortex-ring comparison figures.

set -euo pipefail

cd "$(dirname "$0")"
rm -rf figures
mkdir -p figures

plot() {
    for format in png pdf; do
        python "$@" --format "$format"
    done
}

plot assets/plot_vortex_ring_motion.py
plot assets/plot_vortex_ring_energy.py
plot assets/plot_vortex_ring_circulation.py

echo "Figures saved to figures/"
