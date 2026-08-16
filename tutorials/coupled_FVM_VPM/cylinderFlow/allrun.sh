#!/usr/bin/env bash
# Run the recommended Re=100 hybrid cylinder benchmark.
# This script intentionally takes no options; the complete scientific setup is
# explicit and version-controlled below.
set -euo pipefail

cd "$(dirname "$0")"

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    echo "Edit the OPENONDA_* block in allrun.sh instead of passing options." >&2
    exit 2
fi

export OPENONDA_SMOKE=0
export OPENONDA_T_END=20.0
export OPENONDA_FVM_CORES=4
export OPENONDA_FVM_DT=0.025
export OPENONDA_VPM_DT=0.10
export OPENONDA_SPACING=0.10
export OPENONDA_FVM_DOWNSTREAM_BUFFER=0.50
export OPENONDA_SAMPLE_SPACING=0.10
export OPENONDA_MAX_PARTICLES=1200000

export OPENONDA_FORCE_INTERVAL=0.10
export OPENONDA_DIAGNOSTIC_INTERVAL=1.0
export OPENONDA_CHECKPOINT_INTERVAL=5.0
export OPENONDA_VOLUME_INTERVAL=10.0

echo
echo "===== CLEAN HYBRID ====="
echo
./allclean.sh

echo
echo "===== SIMULATE HYBRID ====="
echo
mkdir -p solution
python -u cylinderFlow_setup.py 2>&1 | tee solution/cylinderFlow.log

echo
echo "===== VALIDATE HYBRID ====="
echo
python assets/check_run.py

echo
echo "===== DONE ====="
echo "Hybrid run completed. Run ./allplot.sh after referenceFlow/allrun.sh."
