#!/bin/bash -e

python setup_les.py --variant baseline
python setup_les.py --variant stretching_viscosity
python setup_les.py --variant p_moments
python setup_les.py --variant splitting
