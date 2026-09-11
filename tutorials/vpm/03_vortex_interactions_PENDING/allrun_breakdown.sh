#!/bin/bash -e

# Separate seeded instability battery. The default allrun.sh remains the
# unperturbed Re_Gamma=3000 kinematic control battery.

python setup_les.py --scenario seeded_breakdown --variant baseline
python setup_les.py --scenario seeded_breakdown --variant stretching_viscosity
python setup_les.py --scenario seeded_breakdown --variant p_moments
python setup_les.py --scenario seeded_breakdown --variant splitting
