#!/usr/bin/env bash
# Make every vortex-ring figure (motion, energy, circulation).
#
# Usage:
#   ./allplot.sh        PNG figures (default)
#   ./allplot.sh pdf    PDF figures
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

# Post-processing: validation and the three figures in both required formats
"$python_bin" assets/postprocess.py --pre-plot
"$python_bin" assets/plot_vortex_ring_motion.p
"$python_bin" assets/plot_vortex_ring_energy.py
"$python_bin" assets/plot_vortex_ring_circulation.py
"$python_bin" assets/plot_vortex_ring_motion.py 
"$python_bin" assets/plot_vortex_ring_energy.py
"$python_bin" assets/plot_vortex_ring_circulation.py
"$python_bin" assets/postprocess.py
