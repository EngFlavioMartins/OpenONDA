#!/bin/bash -e

# Compare the existing h/R0=.06 seeded baseline with the isolated h/R0=.05
# coverage qualification at common native times. These tools read saved native
# HDF5/CSV/VTS outputs and do not reconstruct missing temporal frames.
python assets/plot_core_sections.py \
    --runs cs_breakdown_baseline cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation \
    --times .75 1.5 2.25 2.7 3 \
    --format png \
    --output figures/cs_breakdown_coverage_core_sections
python assets/assess_breakdown.py \
    --runs cs_breakdown_baseline cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation \
    --output figures/cs_breakdown_coverage
python assets/plot_seeded_history.py \
    --runs cs_breakdown_baseline cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation \
    --output figures/cs_breakdown_coverage
python assets/assess_particle_coverage.py \
    --runs cs_breakdown_baseline cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation \
    --output-directory figures/cs_breakdown_coverage
python assets/plot_cross_sections.py \
    --runs cs_breakdown_baseline cs_breakdown_coverage_h05_fixed_sigma_cpu_t6_continuation \
    --time 2.7 \
    --output figures/cs_breakdown_coverage_cross_sections/cross_sections_seeded_breakdown_t2.7.png
