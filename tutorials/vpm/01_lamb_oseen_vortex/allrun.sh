#!/bin/bash
set -e

cd "$(dirname "$0")"

./allclean.sh

python setup.py vortex CS
python -m assets.rwm_ensemble vortex --number-of-realizations 10
python setup.py vortex DVH
python setup.py vortex GBD
python setup.py dipole CS
python -m assets.rwm_ensemble dipole --number-of-realizations 10
python setup.py dipole DVH
python setup.py dipole GBD
python setup.py merging CS
python -m assets.rwm_ensemble merging --number-of-realizations 10
python setup.py merging DVH
python setup.py merging GBD
python -m assets.postprocess --aggregate-rwm
