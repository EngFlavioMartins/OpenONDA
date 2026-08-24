#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

format="${1:-png}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;;
esac

rm -rf figures
mkdir figures

python assets/validate_plot_inputs.py
python assets/plot_velocity_profiles.py --format "$format"
python assets/plot_velocity_fields.py --format "$format"
python assets/plot_coupling_diagnostics.py --format "$format"

echo "Figures are complete in $PWD/figures"
