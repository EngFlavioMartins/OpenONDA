#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

# Steady validation case: Re = 30 cylinder (Constant et al. 2017).
# Quality targets: Cd = 1.74-1.80, recirculation L/D = 1.55-1.70.
# For the unsteady von Karman validation run instead:
#   python cylinderIBM_setup.py --Re 100 --end-time 150 --h 0.05
#   ./allplot.sh --Re 100
python cylinderIBM_setup.py --Re 30 --end-time 60 --h 0.0625 \
    --write-interval-time 5.0 2>&1 | tee solution/cylinderIBM.log

./allplot.sh --Re 30
echo
echo "All runs and plots complete."
