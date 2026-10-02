#!/bin/bash
set -e
# Usage: ./allplot.sh [both|png|pdf] (default: both)

cd -- "$(dirname -- "$0")"

python -m openonda.results restore

python assets/postprocess.py
python assets/plot_velocity_profiles.py --format "${1:-both}"
python assets/plot_coupled_fvm_vpm_fields.py --format "${1:-both}"
python assets/plot_reference_fvm_vpm_fields.py --format "${1:-both}"
python assets/plot_reference_fvm_coupled_fvm_fields.py --format "${1:-both}"
python assets/postprocess.py --report
python assets/plot_coupling_diagnostics.py --format "${1:-both}"
