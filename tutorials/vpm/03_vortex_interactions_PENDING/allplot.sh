#!/bin/bash -e

python assets/plot_core_sections.py --runs cs_baseline cs_stretching_viscosity cs_p_moments cs_splitting --times 1.5 3.3 4.5 6 7.5 9 --format png
python assets/assess_lbm_agreement.py cs_baseline cs_stretching_viscosity cs_p_moments cs_splitting --output figures/cs
