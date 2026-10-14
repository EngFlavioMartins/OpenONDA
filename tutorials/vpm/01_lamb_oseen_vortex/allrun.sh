#!/bin/bash
set -e

cd "$(dirname "$0")"

./allclean.sh

python setup.py vortex cs
python -m assets.rwm_ensemble vortex --number-of-realizations 10
python setup.py vortex dvh
python setup.py vortex gbd
python setup.py dipole cs
python -m assets.rwm_ensemble dipole --number-of-realizations 10
python setup.py dipole dvh
python setup.py dipole gbd
python setup.py merging cs
python -m assets.rwm_ensemble merging --number-of-realizations 10
python setup.py merging dvh
python setup.py merging gbd
python -m assets.postprocess --aggregate-rwm
