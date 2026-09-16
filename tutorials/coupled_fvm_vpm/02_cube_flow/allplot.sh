#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)

cd -- "$(dirname -- "$0")"
python assets/postprocess.py
python assets/plot_velocity_profiles.py --format "${1:-png}"
python assets/plot_coupled_fvm_vpm_fields.py --format "${1:-png}"
python assets/plot_reference_fvm_vpm_fields.py --format "${1:-png}"
python assets/plot_reference_fvm_coupled_fvm_fields.py --format "${1:-png}"
python assets/plot_coupling_diagnostics.py --format "${1:-png}"
python assets/plot_reference_force_history.py --format "${1:-png}"
