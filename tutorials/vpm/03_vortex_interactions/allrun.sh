#!/bin/bash -e
cd -- "$(dirname -- "$0")"

./allclean.sh
python setup.py baseline
python setup.py selective_eddy_viscosity
python setup.py pedrizzetti_relaxation
python setup.py particle_splitting
