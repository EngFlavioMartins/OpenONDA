#!/usr/bin/env bash
# Run the recommended production-resolution overlap-filter validation case.
# This script intentionally takes no options: edit this configuration block so
# a scientific run's complete setup is visible and version-controlled here.
set -euo pipefail

cd "$(dirname "$0")"

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    echo "Edit the explicit OPENONDA_* configuration in allrun.sh instead of passing options." >&2
    exit 2
fi

# Recommended next investigation: isolate the face-aware particle filter at
# the validated RK2/0.05 s coupling cadence.  t=2.4 s gives four coincident
# field frames (0.6, 1.2, 1.8, 2.4) before committing to the t=6 wake run.
export OPENONDA_SMOKE=0
export OPENONDA_T_END=2.4
export OPENONDA_FVM_CORES=4
export OPENONDA_DT_VPM=0.05
export OPENONDA_VPM_SCHEME=RK2

export OPENONDA_SPACING=0.04
export OPENONDA_PARTICLE_SPACING=0.04
export OPENONDA_FVM_CELL_SIZE=0.04
export OPENONDA_SAMPLE_SPACING=0.04
export OPENONDA_SURFACE_CELL_SIZE=0.015
export OPENONDA_MAX_PARTICLES=1500000

export OPENONDA_OVERLAP_SHELL_PRUNE_MULTIPLIER=10.0
export OPENONDA_FORCE_INTERVAL=0.15
export OPENONDA_DIAGNOSTIC_INTERVAL=0.60
export OPENONDA_CHECKPOINT_INTERVAL=1.0
export OPENONDA_VOLUME_INTERVAL=1.0

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
mkdir -p solution
python -u cubeFlow_setup.py 2>&1 | tee solution/cubeFlow.log
