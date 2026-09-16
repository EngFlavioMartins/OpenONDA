#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf|both] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_core_sections.py --runs baseline selective_eddy_viscosity pedrizzetti_relaxation particle_splitting --times 1.5 3 3.3 4.5 6 7.5 9 --format "${1:-png}" --output figures --auxiliary-output figures/auxiliary --clean-output
python assets/plot_core_trajectories.py baseline selective_eddy_viscosity pedrizzetti_relaxation particle_splitting --merge-bridge .9 --format "${1:-png}" --output figures --auxiliary-output figures/auxiliary
python assets/plot_diagnostic_histories.py baseline selective_eddy_viscosity pedrizzetti_relaxation particle_splitting --format "${1:-png}" --output figures
python assets/plot_group_history.py baseline selective_eddy_viscosity pedrizzetti_relaxation particle_splitting --format "${1:-png}" --output figures
