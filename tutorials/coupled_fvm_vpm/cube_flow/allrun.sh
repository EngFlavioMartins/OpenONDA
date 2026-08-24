#!/usr/bin/env bash
# Run the coupled cube-flow case.
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
mkdir -p solution
python -u cube_flow_setup.py 2>&1 | tee solution/cube_flow.log
