#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.tutorial_runner . setup baseline
python -m openonda.tutorial_runner . setup selective_eddy_viscosity
python -m openonda.tutorial_runner . setup pedrizzetti_relaxation
python -m openonda.tutorial_runner . setup particle_splitting
