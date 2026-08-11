#!/usr/bin/env bash
# Generate the delta-wing figures.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p figures

plot() {
    for format in png pdf; do
        python "$@" --format "$format"
    done
}

plot assets/plot_delta_wing_forces.py
plot assets/plot_delta_wing_circulation_history.py

echo "Figures saved to figures/"
