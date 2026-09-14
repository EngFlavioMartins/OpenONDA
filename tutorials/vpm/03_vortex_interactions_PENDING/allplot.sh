#!/bin/bash -e

python assets/plot_core_sections.py --runs fig5_baseline fig5_stretching_viscosity fig5_p_moments --times 1.5 3 3.3 4.5 6 7.5 9 --include-final --format png --output figures/leapfrogging_study/core_sections
python assets/assess_lbm_agreement.py fig5_baseline fig5_stretching_viscosity fig5_p_moments --peak-merge-bridge .9 --output figures/leapfrogging_study
