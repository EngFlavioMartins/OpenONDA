#!/usr/bin/env bash
# Run the coupled cylinder-shedding case.
#
# Usage:
#   ./allrun.sh
#   OPENONDA_GRID=g1 ./allrun.sh
#   OPENONDA_SMOKE=1 ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh
mkdir -p solution

echo
echo "===== SIMULATE (CUT-CELL FVM-VPM) ====="
echo
echo "Output: $PWD/solution and $PWD/samples"
python -u cylinder_shedding_flow_setup.py 2>&1 | tee solution/cylinder_shedding_flow.stdout.log

echo
echo "===== DONE ====="
echo
echo "Simulation completed. Run ./allplot.sh for figures or ./allvalidate.sh for the full gate."
