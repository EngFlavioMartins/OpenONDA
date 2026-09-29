#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py vortex CS
python assets/rwm_ensemble.py vortex --number-of-realizations 10 --converge
python setup.py vortex DVH
python setup.py vortex GBD

python setup.py dipole CS
python assets/rwm_ensemble.py dipole --number-of-realizations 10 --converge
python setup.py dipole DVH
python setup.py dipole GBD

python setup.py merging CS
python assets/rwm_ensemble.py merging --number-of-realizations 10 --converge
python setup.py merging DVH
python setup.py merging GBD
