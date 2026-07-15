#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# Scalar step advection at Co = 0.5: convection-scheme sharpness/boundedness
# benchmark against the exact translated step (validation is analytic).
python stepProfile_setup.py --end-time 0.5 --nx 100 \
    --schemes upwind,limitedLinear,superbee 2>&1 | tee solution/stepProfile.log

./allplot.sh
echo
echo "All runs and plots complete."
