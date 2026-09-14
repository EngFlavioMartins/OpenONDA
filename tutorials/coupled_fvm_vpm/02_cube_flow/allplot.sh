#!/bin/bash -e

set -e
cd -- "$(dirname -- "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OPENONDA_CUBE_REFERENCE_SAMPLES="$PWD/reference_flow/samples/fine"
python assets/prepare_fine_reference.py
python assets/plot_velocity_profiles.py --format "${1:-png}"
python assets/plot_velocity_fields.py --format "${1:-png}"
python assets/plot_coupling_diagnostics.py --format "${1:-png}"
