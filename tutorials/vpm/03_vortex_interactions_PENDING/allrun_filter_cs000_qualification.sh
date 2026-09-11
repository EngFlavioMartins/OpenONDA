#!/bin/bash

# Seeded Re_Gamma=3415 molecular-CS contrast through t=0.6 s.
python3 setup_les.py \
    --variant baseline \
    --scenario seeded_breakdown \
    --compute-device CPU \
    --qualification \
    --steps 80 \
    --wall-minutes 5 \
    --particle-spacing .06 \
    --particle-core-radius .06 \
    --smagorinsky 0 \
    --case-name cs_breakdown_filter_cs000_cpu_t6_step080
