#!/bin/bash -e

python assets/postprocess.py --extract-fields
python assets/plot_vortex_comparison.py --format both
python assets/plot_dipole_comparison.py --format both
python assets/plot_merging_comparison.py --format both
python assets/plot_vortex_surface_fields.py --format both
python assets/plot_lamboseen_energy.py --format both
python assets/plot_merging_snapshots.py --format both
