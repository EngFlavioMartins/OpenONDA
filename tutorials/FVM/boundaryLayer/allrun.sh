#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# Laminar flat plate at Re_L = 1e4 (mesh built in-memory).
# Validation: Blasius profile u/U = f'(eta) and Cf = 0.664/sqrt(Re_x)
# (Blasius 1908; Schlichting, Boundary-Layer Theory).
python boundaryLayer_setup.py --Re 1e4 --end-time 8 2>&1 | tee solution/boundaryLayer.log

./allplot.sh --Re 1e4
echo
echo "All runs and plots complete."
