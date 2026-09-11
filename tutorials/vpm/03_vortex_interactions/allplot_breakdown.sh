#!/bin/bash -e

# Native seeded outputs currently available: baseline and stretching_viscosity.
# p_moments and splitting remain unrun and are intentionally not invented here.
python assets/plot_core_sections.py --runs cs_breakdown_baseline cs_breakdown_stretching_viscosity --times 1.5 3.3 4.5 6 7.5 9 --format png --output figures/cs_breakdown_core_sections
python assets/assess_breakdown.py --runs cs_breakdown_baseline cs_breakdown_stretching_viscosity --output figures/cs_breakdown
python assets/plot_seeded_history.py --runs cs_breakdown_baseline cs_breakdown_stretching_viscosity --output figures/cs_breakdown

# The available LBM trajectory is the distinct unperturbed Re_Gamma=3000
# kinematic reference; do not score this seeded Re=3415 battery against it.
