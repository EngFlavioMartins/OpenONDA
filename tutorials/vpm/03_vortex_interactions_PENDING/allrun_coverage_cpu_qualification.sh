#!/bin/bash -e

# Two-thread CPU cost/startup qualification for the isolated coverage case.
# This short run is not a breakdown or dynamic-coverage comparison.
export TI_CPU_MAX_NUM_THREADS=2
python setup_les.py \
    --variant baseline \
    --scenario seeded_breakdown \
    --compute-device CPU \
    --qualification \
    --steps 40 \
    --wall-minutes 10 \
    --particle-spacing .05 \
    --particle-core-radius .06 \
    --smagorinsky .24080542149013215 \
    --case-name cs_breakdown_coverage_h05_fixed_sigma_cpu_qualification
