#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# Laminar backward-facing-step flow at Re_h=100. The mesh contains the solid
# upstream block and vertical step; this is not a scalar step-profile case.
python stepProfile_setup.py --Re 100 --end-time 12 2>&1 | tee solution/stepProfile.log

./allplot.sh
echo
echo "All runs and plots complete."
