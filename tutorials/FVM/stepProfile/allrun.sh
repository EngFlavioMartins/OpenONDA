#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# Laminar backward-facing-step flow at Re_h = U_b h / nu = 75, i.e.
# Re = U_b (2h) / nu = 150 in the Armaly et al. (1983) convention, where the
# measured primary reattachment is x1/S = 4.2.  The mesh contains the solid
# upstream block and vertical step; this is not a scalar step-profile case.
python stepProfile_setup.py --Re 75 --end-time 40 2>&1 | tee solution/stepProfile.log

./allplot.sh --Re 75
echo
echo "All runs and plots complete."
