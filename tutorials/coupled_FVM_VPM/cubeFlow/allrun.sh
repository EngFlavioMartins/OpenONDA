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

# Recommended compact production case.  Transfer both the VPM velocity and its
# momentum-equation pressure gradient to the FVM numerical boundary.
export OPENONDA_SMOKE=0
export OPENONDA_T_END=2.4
export OPENONDA_FVM_CORES=4
export OPENONDA_DT_VPM=0.05
export OPENONDA_VPM_SCHEME=RK2
export OPENONDA_DONOR_BOUNDARY_MODE=pressure_gradient
export OPENONDA_FVM_OUTER_PADDING=0.30

export OPENONDA_SPACING=0.04
export OPENONDA_PARTICLE_SPACING=0.04
export OPENONDA_FVM_CELL_SIZE=0.038
export OPENONDA_FVM_OUTER_CELL_SIZE=0.076
export OPENONDA_SAMPLE_SPACING=0.04
export OPENONDA_SURFACE_CELL_SIZE=0.015
export OPENONDA_FVM_XMAX=1.5
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
