#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.tutorial_runner . assets.plot_coupling_diagnostics --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_reference_forces --format "${1:-both}"
python -m openonda.tutorial_runner . assets.plot_velocity_profiles --format "${1:-both}"
