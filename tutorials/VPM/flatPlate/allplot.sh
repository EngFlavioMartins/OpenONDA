#!/usr/bin/env bash
# Generate every flat-plate comparison figure.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p figures

plot() {
    for format in png pdf; do
        python "$@" --format "$format"
    done
}

plot assets/plot_plate_polar.py
plot assets/plot_plate_staticvsmoving.py
plot assets/plot_plate_spanwise.py
plot assets/plot_flat_plate_kelvin.py

echo "Figures saved to figures/"
