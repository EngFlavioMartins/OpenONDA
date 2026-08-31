#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh                                          # wipe previous solution/, samples/, figures/
mkdir -p solution

# DNS cases: no subgrid model; three vortex-stretching operators
echo "===== dns_direct ====="                           # direct stretching
"$python_bin" -u setup.py --variant dns_direct

echo "===== dns_transposed ====="                       # transposed stretching — Kelvin circulation preservation
"$python_bin" -u setup.py --variant dns_transposed

echo "===== dns_mixed ====="                            # mixed direct/transposed stretching
"$python_bin" -u setup.py --variant dns_mixed

# LES: transposed stretching + Smagorinsky (C_s=0.20)
echo "===== les_transposed ====="                       # subgrid closure for under-resolved turbulence
"$python_bin" -u setup.py --variant les_transposed
