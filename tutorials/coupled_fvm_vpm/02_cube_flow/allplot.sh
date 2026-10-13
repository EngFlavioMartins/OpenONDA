#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_forces
python -m assets.plot_velocity_profiles
python -m assets.postprocess
python -m assets.plot_coupled_fvm_vpm_fields --format "${1:-both}"
python -m assets.plot_reference_fvm_vpm_fields --format "${1:-both}"
python -m assets.plot_reference_fvm_coupled_fvm_fields --format "${1:-both}"
python -m assets.plot_coupling_diagnostics --format "${1:-both}"
