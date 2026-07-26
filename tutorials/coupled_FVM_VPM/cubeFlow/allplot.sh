#!/bin/sh
cd "$(dirname "$0")" || exit 1
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

python assets/plot_velocity_profiles.py "$@"
python assets/plot_velocity_fields.py "$@"
python assets/plot_wake_errors.py "$@"
python assets/plot_forces.py "$@"