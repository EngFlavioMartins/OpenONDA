#!/bin/bash -e

if [ "$#" -gt 0 ]; then
    runs=("$@")
    output=figures/leapfrogging_study
    sections="$output/core_sections"
    times=(1.5 3 3.3 4.5 6 7.5 9)
    final_option=(--include-final)
else
    runs=(cs_baseline cs_stretching_viscosity cs_p_moments cs_splitting)
    output=figures/cs
    sections=figures/core_sections
    times=(1.5 3.3 4.5 6 7.5 9)
    final_option=()
fi

python assets/plot_core_sections.py --runs "${runs[@]}" --times "${times[@]}" "${final_option[@]}" --format png --output "$sections"
python assets/assess_lbm_agreement.py "${runs[@]}" --output "$output"
