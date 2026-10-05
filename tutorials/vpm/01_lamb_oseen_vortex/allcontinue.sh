#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.tutorial_runner . setup vortex CS
python -m openonda.tutorial_runner . assets.rwm_ensemble vortex --number-of-realizations 10
python -m openonda.tutorial_runner . setup vortex DVH
python -m openonda.tutorial_runner . setup vortex GBD

python -m openonda.tutorial_runner . setup dipole CS
python -m openonda.tutorial_runner . assets.rwm_ensemble dipole --number-of-realizations 10
python -m openonda.tutorial_runner . setup dipole DVH
python -m openonda.tutorial_runner . setup dipole GBD

python -m openonda.tutorial_runner . setup merging CS
python -m openonda.tutorial_runner . assets.rwm_ensemble merging --number-of-realizations 10
python -m openonda.tutorial_runner . setup merging DVH
python -m openonda.tutorial_runner . setup merging GBD

python -m openonda.tutorial_runner . assets.postprocess --aggregate-rwm
