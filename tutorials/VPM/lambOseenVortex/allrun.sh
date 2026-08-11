#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark and make its comparison figures.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

# Output periods [s].
SAMPLE_PERIOD=0.25
BACKUP_PERIOD=5.0

run() {
    python vortex_setup.py "$1" "$2" "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
    ./allplot.sh -png
}

for physics in vortex dipole merging; do
    for scheme in cs rwm dvh gbd; do
        run "$physics" "$scheme"
    done
done
