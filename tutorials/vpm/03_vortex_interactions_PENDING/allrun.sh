#!/bin/bash -e
cd -- "$(dirname -- "$0")"
./allclean.sh

python setup.py baseline
python setup.py stretching_viscosity
python setup.py p_moments
