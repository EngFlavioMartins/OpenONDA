#!/bin/sh
cd "$(dirname "$0")" || exit 1
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

python assets/plot_velocity_profiles.py "$@"
python assets/plot_velocity_fields.py "$@"
python assets/plot_wake_errors.py "$@"
python assets/plot_compare_profiles_frames.py "$@"

python assets/plot_diagnostics.py
python assets/plot_forces.py
python assets/plot_wake.py
python assets/plot_audit_coupling.py
python assets/plot_donor_decomposition.py
python assets/plot_compare_centerline.py
python assets/plot_compare_profiles.py
python assets/plot_compare_interface.py
python assets/plot_compare_forces.py
python assets/plot_waves_hovmoller.py
python assets/plot_waves_amplitude.py
python assets/plot_waves_snapshot.py
