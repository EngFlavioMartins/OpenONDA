#!/usr/bin/env bash
cd "$(dirname "$0")" && ./allclean.sh
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous-scheme CS  --case-name vortex_cs
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous-scheme RWM --case-name vortex_rwm
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous-scheme DVH --case-name vortex_dvh
python -u lambossen_setup.py --gamma1 +1 --gamma2  0 --viscous-scheme GBD --case-name vortex_gbd
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous-scheme CS  --case-name dipole_cs
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous-scheme RWM --case-name dipole_rwm
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous-scheme DVH --case-name dipole_dvh
python -u lambossen_setup.py --gamma1 +1 --gamma2 -1 --viscous-scheme GBD --case-name dipole_gbd
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous-scheme CS  --case-name merging_cs
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous-scheme RWM --case-name merging_rwm
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous-scheme DVH --case-name merging_dvh
python -u lambossen_setup.py --gamma1 +1 --gamma2 +1 --viscous-scheme GBD --case-name merging_gbd
./allplot.sh
