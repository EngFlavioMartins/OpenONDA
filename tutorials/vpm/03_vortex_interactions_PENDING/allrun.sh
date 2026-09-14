#!/bin/bash
./allclean.sh || exit 1

python setup.py baseline
python setup.py stretching_viscosity
python setup.py p_moments
