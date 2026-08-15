#!/usr/bin/env bash
# Run the coupled cube-flow case and check it.
# Usage: ./allrun.sh [cubeFlow_setup.py options]
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
python -u cubeFlow_setup.py "$@" 2>&1 | tee solution/cubeFlow.log
