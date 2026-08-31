#!/usr/bin/env bash
# Run the fully meshed FVM reference for the cylinder-shedding experiment.
#
# Usage:
#   ./allrun.sh
#   OPENONDA_SMOKE=1 ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"
case_dir="${OPENONDA_REFERENCE_CASE_DIR:-$PWD}"
if [ -z "${OPENONDA_RESTART_FROM:-}" ]; then
    if [ "$case_dir" = "$PWD" ]; then
        ./allclean.sh
    else
        mkdir -p "$case_dir"
    fi
    tee_flag=()
else
    echo "Restarting from $OPENONDA_RESTART_FROM; retaining existing solution and samples."
    tee_flag=(-a)
fi
mkdir -p "$case_dir/solution"

cores="${OPENONDA_FVM_CORES:-6}"
solver=(python -u reference_flow_setup.py)
if [ "$cores" -gt 1 ]; then
    solver=(mpiexec --bind-to none -n "$cores" python -u reference_flow_setup.py)
fi

echo
echo "===== SIMULATE (CUT-CELL FVM REFERENCE) ====="
echo
echo "Output: $case_dir/solution and $case_dir/samples"
"${solver[@]}" 2>&1 | tee "${tee_flag[@]}" "$case_dir/solution/reference_flow.stdout.log"

echo
echo "===== DONE ====="
echo
echo "Reference simulation completed. Run ../allvalidate.sh for the full independence gate."
