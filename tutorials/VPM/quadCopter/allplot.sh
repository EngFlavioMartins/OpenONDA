#!/usr/bin/env bash
# Generate the quadcopter diagnostic figures.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p figures

plot() {
    for format in png pdf; do
        python "$@" --format "$format"
    done
}

plot assets/plot_quadcopter_particle_count.py
plot assets/plot_quadcopter_vorticity_history.py

echo "Figures saved to figures/"
