#!/usr/bin/env bash
# Run the moving- and fixed-plate angle-of-attack sweeps.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

SAMPLE_PERIOD=0.0625
BACKUP_PERIOD=0.05  # One animation frame per half-chord of flow evolution.

run() {
    python setup_plate.py "$1" "$2" "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
}

for angle in -10 -5 -2 0 2 5 8 10 12 15; do
    run moving "$angle"
done

for angle in -10 -5 -2 0 2 5 8 10 12 15; do
    run static "$angle"
done

./allplot.sh
