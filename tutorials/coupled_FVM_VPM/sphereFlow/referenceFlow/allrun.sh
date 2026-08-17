#!/usr/bin/env bash
# Fully meshed reference for the Re = 300 sphere benchmark.
set -euo pipefail
cd "$(dirname "$0")"
export OPENONDA_SMOKE=0
export OPENONDA_SPACING=0.0625
export OPENONDA_FVM_DT=0.02
export OPENONDA_T_END=100.0
export OPENONDA_FVM_CORES=1
export OPENONDA_FORCE_INTERVAL=0.1
export OPENONDA_DIAGNOSTIC_INTERVAL=1.0
export OPENONDA_VOLUME_INTERVAL=25.0

echo; echo "===== CLEAN REFERENCE ====="; echo
./allclean.sh
echo; echo "===== SIMULATE REFERENCE ====="; echo
mkdir -p solution
python -u referenceFlow_setup.py 2>&1 | tee solution/referenceFlow.log
echo; echo "===== DONE ====="
