#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.postprocess --extract-fields
python -m assets.plot_vortex_comparison
python -m assets.plot_dipole_comparison
python -m assets.plot_merging_comparison
python -m assets.plot_vortex_surface_fields
python -m assets.plot_lamboseen_energy
python -m assets.plot_merging_snapshots
