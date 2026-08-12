#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark and make its comparison figures.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

# Output periods [s].
VORTEX_SAMPLE_PERIOD=1.0
DIPOLE_SAMPLE_PERIOD=2.5
MERGING_SAMPLE_PERIOD=1.25
BACKUP_PERIOD=40.0

run() {
    python vortex_setup.py "$1" "$2" "$3" "$BACKUP_PERIOD"
    python assets/validate_results.py "$1" "$2"
    ./allplot.sh -png
}

for physics in vortex dipole merging; do
    case "$physics" in
        vortex) sample_period="$VORTEX_SAMPLE_PERIOD" ;;
        dipole) sample_period="$DIPOLE_SAMPLE_PERIOD" ;;
        merging) sample_period="$MERGING_SAMPLE_PERIOD" ;;
    esac
    for scheme in cs rwm dvh gbd; do
        run "$physics" "$scheme" "$sample_period"
    done
done
