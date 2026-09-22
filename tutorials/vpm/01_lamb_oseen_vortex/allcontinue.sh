#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py vortex CS
python setup.py vortex RWM --ensemble --number-of-realizations 10 --converge
python setup.py vortex DVH
python setup.py vortex GBD

python setup.py dipole CS
python setup.py dipole RWM --ensemble --number-of-realizations 10 --converge
python setup.py dipole DVH
python setup.py dipole GBD

python setup.py merging CS
python setup.py merging RWM --ensemble --number-of-realizations 10 --converge
python setup.py merging DVH
python setup.py merging GBD
