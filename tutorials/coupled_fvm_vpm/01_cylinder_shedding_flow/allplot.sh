#!/bin/bash
set -euo pipefail
# Usage: ./allplot.sh [both|png|pdf]
cd -- "$(dirname -- "$0")"
export MPLBACKEND=Agg
format="${1:-png}"

python assets/plot_coupling_diagnostics.py --format "$format"
python assets/plot_reference_forces.py --format "$format"
python assets/plot_velocity_profiles.py --format "$format"
