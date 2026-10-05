#!/bin/bash
set -e

# Usage: ./allplot.sh [both|png|pdf] (default: both)
cd "$(dirname "$0")"

python -m openonda.results restore

python -m openonda.tutorial_runner . assets.plot_velocity_profiles --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_coupled_fvm_vpm_fields --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_reference_fvm_vpm_fields --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_reference_fvm_coupled_fvm_fields --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_coupling_diagnostics --format "${1:-both}"
