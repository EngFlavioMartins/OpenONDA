#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

python assets/plot_coupling_diagnostics.py --format "${1:-both}"
python assets/plot_reference_forces.py --format "${1:-both}"
python assets/plot_velocity_profiles.py --format "${1:-both}"
