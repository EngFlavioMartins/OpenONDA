#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/postprocess.py --extract-fields
python assets/plot_vortex_comparison.py --format "${1:-png}"
python assets/plot_dipole_comparison.py --format "${1:-png}"
python assets/plot_merging_comparison.py --format "${1:-png}"
python assets/plot_vortex_surface_fields.py --format "${1:-png}"
python assets/plot_lamboseen_energy.py --format "${1:-png}"
python assets/plot_merging_snapshots.py --format "${1:-png}"
