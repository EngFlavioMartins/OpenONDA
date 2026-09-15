#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf|both] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_core_sections.py --runs baseline selective_eddy_viscosity pedrizzetti_relaxation particle_splitting --times 1.5 3 3.3 4.5 6 7.5 9 --include-final --format "${1:-png}" --output figures/leapfrogging_study/core_sections
python assets/plot_lbm_comparison.py baseline selective_eddy_viscosity pedrizzetti_relaxation particle_splitting --peak-merge-bridge .9 --format "${1:-png}" --output figures/leapfrogging_study
