#!/bin/sh
# Run the native FVM+VPM cube case end to end: clean, solve, plot.
#
# System-agnostic — no OpenFOAM, no cfMesh, no MPI.  The mesh is built in
# memory by cube_setup.py and the cube is an immersed body.  Set HYBRID_T_END
# to shorten the run (e.g. HYBRID_T_END=2.0 ./allrun.sh for a ~4 min preview;
# the default t_end=7.5 develops the full shed wake).
cd "$(dirname "$0")" || exit 1

PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null \
        || conda run -n OpenONDA which python 2>/dev/null \
        || command -v python3 || command -v python)"
echo "Python: $PYTHON"

export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH}"

./allclean.sh

echo ""
echo "=== Running coupled FVM+VPM (t_end=${HYBRID_T_END:-7.5} s) ==="
if ! "$PYTHON" -u cube_setup.py; then
    echo "ERROR: coupled run failed. See solution/vpm_solution.log and solution/coupler.log."
    exit 1
fi

echo ""
echo "=== Post-processing ==="
./allplot.sh || echo "  (plotting failed; check solution/ outputs)"

echo ""
echo "Done. Results in solution/, figures in figures/."
