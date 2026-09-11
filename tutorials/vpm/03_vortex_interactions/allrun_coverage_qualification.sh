#!/bin/bash -e

# Isolate dynamic particle coverage while holding the physical Gaussian core,
# initial numerical core, circulation, seed, dt, and strength-weighted RMS
# Smagorinsky length Cs*Delta matched to the h/R0=.06 baseline.
python setup_les.py \
    --variant baseline \
    --scenario seeded_breakdown \
    --compute-device METAL \
    --qualification \
    --steps 400 \
    --particle-spacing .05 \
    --particle-core-radius .06 \
    --smagorinsky .24080542149013215 \
    --case-name cs_breakdown_coverage_h05_fixed_sigma
