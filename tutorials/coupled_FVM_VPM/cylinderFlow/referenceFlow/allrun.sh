#!/usr/bin/env bash
# Run the matched fully meshed Re=100 cylinder reference.
set -euo pipefail

cd "$(dirname "$0")"

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    echo "Edit the OPENONDA_* block in referenceFlow/allrun.sh instead." >&2
    exit 2
fi

export OPENONDA_SMOKE=0
export OPENONDA_T_END=20.0
export OPENONDA_FVM_CORES=4
export OPENONDA_FVM_DT=0.025
export OPENONDA_SPACING=0.10
export OPENONDA_SAMPLE_SPACING=0.10
export OPENONDA_FORCE_INTERVAL=0.10
export OPENONDA_DIAGNOSTIC_INTERVAL=1.0
# Only t=0, 10, and 20 raw FVM volumes are retained. Line/surface/force
# samples remain dense and are sufficient for every plotting script.
export OPENONDA_VOLUME_INTERVAL=10.0

echo
echo "===== CLEAN REFERENCE ====="
echo
./allclean.sh

echo
echo "===== SIMULATE REFERENCE ====="
echo
mkdir -p solution
python -u referenceFlow_setup.py 2>&1 | tee solution/referenceFlow.log

echo
echo "===== DONE ====="
echo "Reference run completed. Return to .. and run ./allrun.sh."
