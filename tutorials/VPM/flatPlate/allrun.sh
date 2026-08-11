#!/usr/bin/env bash
# Run the moving- and fixed-plate angle-of-attack sweeps.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

SAMPLE_PERIOD=0.0625
BACKUP_PERIOD=0.05  # One animation frame per half-chord of flow evolution.

run() {
    python setup_plate.py "$1" "$2" "$3" "$4" "$5" \
        "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
}

run exp_moving_aoan10 ramp body -10 none
run exp_moving_aoan05 ramp body -5 none
run exp_moving_aoan02 ramp body -2 none
run exp_moving_aoa00 ramp body 0 none
run exp_moving_aoa02 ramp body 2 none
run exp_moving_aoa05 ramp body 5 none
run exp_moving_aoa08 ramp body 8 none
run exp_moving_aoa10 ramp body 10 none
run exp_moving_aoa12 ramp body 12 none
run exp_moving_aoa15 ramp body 15 none

run exp_static_aoan10 static wind -10 none
run exp_static_aoan05 static wind -5 none
run exp_static_aoan02 static wind -2 none
run exp_static_aoa00 static wind 0 none
run exp_static_aoa02 static wind 2 none
run exp_static_aoa05 static wind 5 none
run exp_static_aoa08 static wind 8 planes
run exp_static_aoa10 static wind 10 none
run exp_static_aoa12 static wind 12 none
run exp_static_aoa15 static wind 15 none

./allplot.sh
