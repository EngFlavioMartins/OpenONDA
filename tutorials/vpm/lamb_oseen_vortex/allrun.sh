#!/usr/bin/env bash
# Run the Lamb--Oseen vortex, dipole, and merging-vortex comparisons.
set -euo pipefail

cd "$(dirname "$0")"

rm -rf solution samples figures
mkdir -p solution samples figures

# Isolated vortex: Γ₂ = 0
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme CS --case-name vortex_cs
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme DVH --case-name vortex_dvh
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 0 --viscous-scheme GBD --case-name vortex_gbd
python -u lamb_oseen_rwm_setup.py --case vortex --number-of-realizations 8

# Counter-rotating vortex pair: Γ₂ = -Γ₁
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme CS --case-name dipole_cs
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme DVH --case-name dipole_dvh
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 -1 --viscous-scheme GBD --case-name dipole_gbd
python -u lamb_oseen_rwm_setup.py --case dipole --number-of-realizations 12

# Co-rotating vortex pair: Γ₂ = Γ₁
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme CS --case-name merging_cs
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme DVH --case-name merging_dvh
python -u lamb_oseen_setup.py --circulation1 +1 --circulation2 +1 --viscous-scheme GBD --case-name merging_gbd
python -u lamb_oseen_rwm_setup.py --case merging --number-of-realizations 8

# Ensemble means, physical diagnostics, and figures
python assets/postprocess.py --aggregate-rwm --expected-rwm-vortex-members 8 --expected-rwm-dipole-members 12 --expected-rwm-merging-members 8
python assets/postprocess.py --extract-fields
python assets/postprocess.py --pre-plot
python assets/plot_vortex_comparison.py
python assets/plot_dipole_comparison.py
python assets/plot_merging_comparison.py
python assets/plot_vortex_surface_fields.py
python assets/plot_lamboseen_energy.py
python assets/postprocess.py --manifest
python assets/postprocess.py
