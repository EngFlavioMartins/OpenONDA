#!/usr/bin/env bash
# Run the fully meshed FVM reference for the cylinder-shedding experiment.
#
# Usage:
#   ./allrun.sh
#   OPENONDA_SEED_AMPLITUDE=1e-4 ./allrun.sh
#   OPENONDA_SMOKE=1 ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE (FVM reference) ====="
echo
mkdir -p solution
python -u referenceFlow_setup.py 2>&1 | tee solution/referenceFlow.log

echo
echo "===== DONE ====="
echo
echo "Reference simulation completed. Run ../allvalidate.sh for the full analysis."