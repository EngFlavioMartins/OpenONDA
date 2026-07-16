#!/bin/sh
# Run the fully-meshed reference solution (t_end = 20 s by default).
# Set REF_T_END to shorten (e.g. REF_T_END=0.1 for a smoke run).
cd "$(dirname "$0")" || exit 1

PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null \
        || conda run -n OpenONDA which python 2>/dev/null \
        || command -v python3 || command -v python)"
echo "Python: $PYTHON"

export PYTHONPATH="$(cd ../../../.. && pwd):${PYTHONPATH}"

echo "=== Running referenceFlow (t_end=${REF_T_END:-20} s) ==="
if ! REF_OPERATOR_BACKEND="${REF_OPERATOR_BACKEND:-numba}" "$PYTHON" -u reference_setup.py; then
    echo "ERROR: reference run failed. See solution/ outputs."
    exit 1
fi

echo "Done. Results in solution/."
