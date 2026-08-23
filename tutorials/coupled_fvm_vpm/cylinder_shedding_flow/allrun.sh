#!/usr/bin/env bash
# Run the coupled cylinder-shedding case.
#
# Usage:
#   ./allrun.sh                       unseeded validation run
#   OPENONDA_SEED_AMPLITUDE=1e-4 ./allrun.sh
#   OPENONDA_SMOKE=1 ./allrun.sh      short smoke run
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE (hybrid) ====="
echo
mkdir -p solution
python -u cylinder_shedding_flow_setup.py 2>&1 | tee solution/cylinder_shedding_flow.log

echo
echo "===== DONE ====="
echo
echo "Simulation completed. Run ./allvalidate.sh for the full analysis."