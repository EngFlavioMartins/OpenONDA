#!/bin/bash
set -e

cd "$(dirname "$0")"

# Remove previous output and simulate the three reduced velocities from t = 0.
./allclean.sh
python setup.py --reduced-velocity 4 "$@"
python setup.py --reduced-velocity 6 "$@"
python setup.py --reduced-velocity 8 "$@"
