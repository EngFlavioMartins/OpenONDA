#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh                                          # wipe previous solution/, samples/, figures/
mkdir -p solution

# DNS cases: no subgrid model; three vortex-stretching operators
echo "===== dns_direct ====="                           # direct stretching
"$python_bin" -u ring_setup.py --variant dns_direct

echo "===== dns_transposed ====="                       # transposed stretching — Kelvin circulation preservation
"$python_bin" -u ring_setup.py --variant dns_transposed

echo "===== dns_mixed ====="                            # mixed direct/transposed stretching
"$python_bin" -u ring_setup.py --variant dns_mixed

# LES: transposed stretching + Smagorinsky (C_s=0.20)
echo "===== les_transposed ====="                       # subgrid closure for under-resolved turbulence
"$python_bin" -u ring_setup.py --variant les_transposed

# Post-processing: validation and the three figures in both required formats
"$python_bin" assets/postprocess.py --pre-plot
"$python_bin" assets/plot_vortex_ring_motion.py --format png
"$python_bin" assets/plot_vortex_ring_energy.py --format png
"$python_bin" assets/plot_vortex_ring_circulation.py --format png
"$python_bin" assets/plot_vortex_ring_motion.py --format pdf
"$python_bin" assets/plot_vortex_ring_energy.py --format pdf
"$python_bin" assets/plot_vortex_ring_circulation.py --format pdf
"$python_bin" assets/postprocess.py
