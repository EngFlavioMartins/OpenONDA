#!/usr/bin/env bash
# Run the Lamb--Oseen vortex, dipole, and merging-vortex comparisons.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.lamb_oseen_vortex"

rm -rf solution samples figures
mkdir -p solution samples figures

# Isolated vortex: Γ₂ = 0
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 0 --viscous-scheme CS --case-name vortex_cs
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 0 --viscous-scheme DVH --case-name vortex_dvh
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 0 --viscous-scheme GBD --case-name vortex_gbd
python -u -m "${MODULE}.assets.rwm_ensemble" --case vortex --number-of-realizations 8

# Counter-rotating vortex pair: Γ₂ = -Γ₁
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 -1 --viscous-scheme CS --case-name dipole_cs
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 -1 --viscous-scheme DVH --case-name dipole_dvh
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 -1 --viscous-scheme GBD --case-name dipole_gbd
python -u -m "${MODULE}.assets.rwm_ensemble" --case dipole --number-of-realizations 12

# Co-rotating vortex pair: Γ₂ = Γ₁
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 +1 --viscous-scheme CS --case-name merging_cs
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 +1 --viscous-scheme DVH --case-name merging_dvh
python -u -m "${MODULE}.setup" --circulation1 +1 --circulation2 +1 --viscous-scheme GBD --case-name merging_gbd
python -u -m "${MODULE}.assets.rwm_ensemble" --case merging --number-of-realizations 8

# Ensemble means, physical diagnostics, and figures
python -m "${MODULE}.assets.postprocess" --aggregate-rwm --expected-rwm-vortex-members 8 --expected-rwm-dipole-members 12 --expected-rwm-merging-members 8
python -m "${MODULE}.assets.postprocess" --extract-fields
python -m "${MODULE}.assets.postprocess" --pre-plot
python -m "${MODULE}.assets.plot_vortex_comparison"
python -m "${MODULE}.assets.plot_dipole_comparison"
python -m "${MODULE}.assets.plot_merging_comparison"
python -m "${MODULE}.assets.plot_vortex_surface_fields"
python -m "${MODULE}.assets.plot_lamboseen_energy"
python -m "${MODULE}.assets.postprocess" --manifest
python -m "${MODULE}.assets.postprocess"
