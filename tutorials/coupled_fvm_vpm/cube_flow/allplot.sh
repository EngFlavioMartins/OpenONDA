#!/bin/bash -e

python assets/plot_velocity_profiles.py --format "${1:-png}"
python assets/plot_velocity_fields.py --format "${1:-png}"
python assets/plot_coupling_diagnostics.py --format "${1:-png}"
