#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# Square-cylinder von Karman street at Re = 100 (mesh built in-memory).
# Validation: St = 0.140-0.150, mean Cd = 1.45-1.58
# (Okajima 1982; Sohankar et al. 1998; Sen et al. 2011).
python cubeFlow_setup.py --Re 100 --end-time 120 2>&1 | tee solution/cubeFlow.log

./allplot.sh --Re 100
echo
echo "All runs and plots complete."
