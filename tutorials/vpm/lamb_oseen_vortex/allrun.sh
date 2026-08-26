#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh                                          # wipe previous solution/, samples/, figures/
mkdir -p solution

# Three physics: vortex (Γ₂=0, isolated), dipole (Γ₂<0, counter-rotating), merging (Γ₂>0, co-rotating)
echo "===== CS cases ====="                             # Core Spreading — deterministic particle redistribution
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme CS --case-name vortex_cs --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme CS --case-name dipole_cs --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme CS --case-name merging_cs --compute-device METAL

echo "===== RWM cases ====="                            # Random Walk Method — stochastic particle displacement
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme RWM --case-name vortex_rwm --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme RWM --case-name dipole_rwm --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme RWM --case-name merging_rwm --compute-device METAL

echo "===== DVH cases ====="                            # Diffusion via Vorticity Hierarchy — adaptive octree grid
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme DVH --case-name vortex_dvh --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme DVH --case-name dipole_dvh --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme DVH --case-name merging_dvh --compute-device METAL

echo "===== GBD cases ====="                            # Grid-Based Diffusion — fixed Eulerian grid
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme GBD --case-name vortex_gbd --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme GBD --case-name dipole_gbd --compute-device METAL
"$python_bin" -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme GBD --case-name merging_gbd --compute-device METAL

# Post-processing: field plots (png, pdf) and strict integral validation against analytic solution
"$python_bin" assets/postprocess.py --pre-plot
./allplot.sh png
./allplot.sh pdf
"$python_bin" assets/postprocess.py
