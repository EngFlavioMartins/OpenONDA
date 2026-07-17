#!/bin/sh
# Generate all validation figures for the hybrid FVM-VPM cube-flow case.
# Per-time-step: velocity_profiles, velocity_fields, wake_errors, compare_profiles
# Single-shot: diagnostics, forces, wake, audit, donor, reference comparison, waves
# Re-runnable while a simulation is in progress: existing frames are kept
# (pass --force to regenerate everything).
cd "$(dirname "$0")" || exit 1

PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null \
        || conda run -n OpenONDA which python 2>/dev/null \
        || command -v python3 || command -v python)"

# Matplotlib needs a writable config dir in headless/CI environments.
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH}"

# Per-time-step frames
"$PYTHON" assets/plot_velocity_profiles.py "$@"
"$PYTHON" assets/plot_velocity_fields.py "$@"
"$PYTHON" assets/plot_wake_errors.py "$@"
"$PYTHON" assets/plot_compare_profiles_frames.py "$@"

# Single-shot diagnostics
"$PYTHON" assets/plot_diagnostics.py
"$PYTHON" assets/plot_forces.py
"$PYTHON" assets/plot_wake.py
"$PYTHON" assets/plot_audit_coupling.py
"$PYTHON" assets/plot_donor_decomposition.py
"$PYTHON" assets/plot_compare_centerline.py
"$PYTHON" assets/plot_compare_profiles.py
"$PYTHON" assets/plot_compare_interface.py
"$PYTHON" assets/plot_compare_forces.py
"$PYTHON" assets/plot_waves_hovmoller.py
"$PYTHON" assets/plot_waves_amplitude.py
"$PYTHON" assets/plot_waves_snapshot.py

echo "Figures written to figures/"
