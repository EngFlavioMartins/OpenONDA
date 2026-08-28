#!/usr/bin/env bash
# Run the fully meshed FVM reference for the cylinder-shedding experiment.
#
# Usage:
#   ./allrun.sh
#   OPENONDA_SMOKE=1 ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"
if [ -z "${OPENONDA_RESTART_FROM:-}" ]; then
    ./allclean.sh
else
    echo "Restarting from $OPENONDA_RESTART_FROM; retaining existing solution and samples."
fi
mkdir -p solution

echo
echo "===== SIMULATE (CUT-CELL FVM REFERENCE) ====="
echo
echo "Output: $PWD/solution and $PWD/samples"
python -u reference_flow_setup.py 2>&1 | tee solution/reference_flow.stdout.log

echo
echo "===== DONE ====="
echo
echo "Reference simulation completed. Run ../allvalidate.sh for the full independence gate."
