#!/bin/bash -e

cd -- "$(dirname -- "$0")"

python setup_les.py --variant baseline
python setup_les.py --variant stretching_viscosity
python setup_les.py --variant p_moments
python setup_les.py --variant splitting
