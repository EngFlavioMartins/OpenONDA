#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark and make its comparison figures.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

# Output periods [s].  vortex_setup.py converts these to each method's steps.
SAMPLE_PERIOD=1.0
BACKUP_PERIOD=5.0

run() {
    python vortex_setup.py "$1" "$2" "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
}

run vortex cs
run vortex rwm
run vortex dvh
run vortex gbd

run dipole cs
run dipole rwm
run dipole dvh
run dipole gbd

run merging cs
run merging rwm
run merging dvh
run merging gbd

./allplot.sh
