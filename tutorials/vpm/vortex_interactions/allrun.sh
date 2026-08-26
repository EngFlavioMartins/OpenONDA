#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh                                          # wipe previous solution/, samples/, figures/

# Four baselines: co-rotating leapfrogging and counter-rotating colliding rings,
# each with transposed DNS and calibrated transposed LES (C_s=0.20)
echo "===== leapfrog_dns ====="
"$python_bin" -u rings_setup.py --case leapfrog_dns
echo "===== leapfrog_les ====="
"$python_bin" -u rings_setup.py --case leapfrog_les
echo "===== collide_dns ====="
"$python_bin" -u rings_setup.py --case collide_dns
echo "===== collide_les ====="
"$python_bin" -u rings_setup.py --case collide_les

# Stabilized LES: overshoot-gated filament splitting (no remesh, no relaxation)
echo "===== leapfrog_les_stabilized ====="
"$python_bin" -u rings_setup.py --case leapfrog_les_stabilized
echo "===== collide_les_stabilized ====="
"$python_bin" -u rings_setup.py --case collide_les_stabilized

# Post-processing: physics gate and figures (png, pdf)
"$python_bin" assets/postprocess.py
./allplot.sh png
./allplot.sh pdf
