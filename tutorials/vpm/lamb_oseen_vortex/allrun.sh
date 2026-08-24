#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export OPENONDA_COMPUTE_DEVICE="${OPENONDA_COMPUTE_DEVICE:-METAL}"
# The finite-box FFT energy fallback misses the long-range energy of the
# non-zero-circulation vortex column.  Metal benchmarks evaluate the exact
# unbounded integral in about 1.1 s at 100k, 3.2 s at 200k, and 31 s at the
# 500k capacity ceiling, so every calibrated Lamb--Oseen cloud stays on the
# exact diagnostic path despite the deliberate verification cost.
export OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT="${OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT:-500000}"
./allclean.sh
mkdir -p solution

device="${OPENONDA_COMPUTE_DEVICE:-AUTO}"

run_case() {
    local case_name="$1"
    shift
    echo "===== ${case_name} (${device}) ====="
    python -u lamb_oseen_setup.py "$@" --case-name "$case_name" --compute-device "$device" \
        2>&1 | tee "solution/${case_name}.log"
}

run_case vortex_cs --circulation1 +1 --circulation2 0 --viscous-scheme CS
run_case vortex_rwm --circulation1 +1 --circulation2 0 --viscous-scheme RWM
run_case vortex_dvh --circulation1 +1 --circulation2 0 --viscous-scheme DVH
run_case vortex_gbd --circulation1 +1 --circulation2 0 --viscous-scheme GBD
run_case dipole_cs --circulation1 +1 --circulation2 -1 --viscous-scheme CS
run_case dipole_rwm --circulation1 +1 --circulation2 -1 --viscous-scheme RWM
run_case dipole_dvh --circulation1 +1 --circulation2 -1 --viscous-scheme DVH
run_case dipole_gbd --circulation1 +1 --circulation2 -1 --viscous-scheme GBD
run_case merging_cs --circulation1 +1 --circulation2 +1 --viscous-scheme CS
run_case merging_rwm --circulation1 +1 --circulation2 +1 --viscous-scheme RWM
run_case merging_dvh --circulation1 +1 --circulation2 +1 --viscous-scheme DVH
run_case merging_gbd --circulation1 +1 --circulation2 +1 --viscous-scheme GBD

python assets/validate_results.py --pre-plot
./plot_all.sh png
./plot_all.sh pdf
python assets/validate_results.py
