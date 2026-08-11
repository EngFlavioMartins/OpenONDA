#!/usr/bin/env bash
# Generate the rotor performance, wake, and loading figures.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p figures

plot() {
    for format in png pdf; do
        python "$@" --format "$format"
    done
}

plot assets/plot_rotor_performance.py
plot assets/plot_rotor_wake_planes.py
plot assets/plot_rotor_loading_validation.py

echo "Figures saved to figures/"
