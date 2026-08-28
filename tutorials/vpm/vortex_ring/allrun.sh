#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh                                          # wipe previous solution/, samples/, figures/
mkdir -p solution

# DNS: no subgrid model; three vortex-stretching operators
echo "===== dns_direct ====="                           # direct stretching — reference standard
"$python_bin" -u ring_setup.py --variant dns_direct

echo "===== dns_transposed ====="                       # transposed stretching — Kelvin circulation preservation
"$python_bin" -u ring_setup.py --variant dns_transposed

echo "===== dns_mixed ====="                            # mixed (direct + transposed) — Galilean invariant
"$python_bin" -u ring_setup.py --variant dns_mixed

# LES: transposed stretching + Smagorinsky (C_s=0.20)
echo "===== les_transposed ====="                       # subgrid closure for under-resolved turbulence
"$python_bin" -u ring_setup.py --variant les_transposed

# Post-processing: field plots (png, pdf) and strict integral validation
"$python_bin" assets/postprocess.py --pre-plot
./allplot.sh png
./allplot.sh pdf
"$python_bin" assets/postprocess.py
