#!/bin/bash -e

cd -- "$(dirname -- "$0")"

python assets/plot_core_sections.py --runs les_baseline les_stretching_viscosity les_p_moments les_splitting --times 1.5 3.3 4.5 6 7.5 9 --format png
python assets/assess_lbm_agreement.py les_baseline les_stretching_viscosity les_p_moments les_splitting --output figures/les
