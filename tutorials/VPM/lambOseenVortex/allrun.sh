#!/usr/bin/env bash
cd "$(dirname "$0")" && ./allclean.sh
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous_scheme CS  --case_name vortex_cs
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous_scheme RWM --case_name vortex_rwm
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous_scheme DVH --case_name vortex_dvh
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous_scheme GBD --case_name vortex_gbd
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous_scheme CS  --case_name dipole_cs
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous_scheme RWM --case_name dipole_rwm
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous_scheme DVH --case_name dipole_dvh
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous_scheme GBD --case_name dipole_gbd
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous_scheme CS  --case_name merging_cs
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous_scheme RWM --case_name merging_rwm
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous_scheme DVH --case_name merging_dvh
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous_scheme GBD --case_name merging_gbd
./allplot.sh
