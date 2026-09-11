#!/bin/bash

# Seeded Re_Gamma=3415 Smagorinsky control through t=0.6 s.
python3 setup_les.py \
    --variant baseline \
    --scenario seeded_breakdown \
    --compute-device CPU \
    --qualification \
    --steps 80 \
    --wall-minutes 5 \
    --particle-spacing .06 \
    --particle-core-radius .06 \
    --smagorinsky .20 \
    --case-name cs_breakdown_filter_cs020_cpu_t6_step080
