#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)

cd -- "$(dirname -- "$0")"
python assets/prepare_fine_reference.py
python assets/validate_plot_inputs.py
python assets/plot_velocity_profiles.py --format "${1:-png}"
python assets/plot_velocity_fields.py --format "${1:-png}"
python assets/plot_coupling_diagnostics.py --format "${1:-png}"
python assets/audit_comparison.py --format "${1:-png}"
