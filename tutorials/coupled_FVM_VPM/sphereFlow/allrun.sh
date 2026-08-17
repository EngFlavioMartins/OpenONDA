#!/usr/bin/env bash
# Hybrid FVM-VPM sphere at Re = 300 -- the unsteady coupling benchmark.
# This script takes no options: edit the explicit OPENONDA_* block so a run's
# complete configuration is visible and version-controlled here.
set -euo pipefail
cd "$(dirname "$0")"
if (( $# != 0 )); then
    echo "Usage: $0" >&2
    echo "Edit the OPENONDA_* block in allrun.sh instead of passing options." >&2
    exit 2
fi

export OPENONDA_SMOKE=0
# h/D = 1/16.  tests/coupler/test_handoff_convergence shows the hand-off needs
# ~4 lattice cells per vortex-core radius for 1%; the hairpin cores two
# diameters downstream are ~0.25 D.
export OPENONDA_SPACING=0.0625
export OPENONDA_FVM_DT=0.02
export OPENONDA_VPM_DT=0.10
# Shedding is established by tU/D ~ 30; 100 units leaves ~9 periods of
# statistics after the SHEDDING_START cut.
export OPENONDA_T_END=100.0
export OPENONDA_SHEDDING_START=40.0
export OPENONDA_FVM_DOWNSTREAM_BUFFER=0.5
# Serial.  petsc_replicated keeps a full Python+numba+PETSc runtime on every
# rank (measured 1.52 GB/rank on the 86k-cell cylinder reference, 6.09 GB over
# four) for a mesh whose numerical data is a tenth of that.
export OPENONDA_FVM_CORES=1
export OPENONDA_MAX_PARTICLES=3000000

export OPENONDA_FORCE_INTERVAL=0.1
export OPENONDA_DIAGNOSTIC_INTERVAL=1.0
export OPENONDA_CHECKPOINT_INTERVAL=10.0
export OPENONDA_VOLUME_INTERVAL=25.0

echo; echo "===== CLEAN HYBRID ====="; echo
./allclean.sh

echo; echo "===== SIMULATE HYBRID ====="; echo
mkdir -p solution
python -u sphereFlow_setup.py 2>&1 | tee solution/sphereFlow.log

echo; echo "===== DONE ====="
echo "Now run referenceFlow/allrun.sh, then ./allplot.sh."
