#!/usr/bin/env bash
# Make every vortex-ring figure (motion, energy, circulation).
#
# Usage: ./allplot.sh
set -euo pipefail

cd "$(dirname "$0")"

# Post-processing: validation and the three figures in both required formats
python assets/postprocess.py --pre-plot
for figure_format in png pdf; do
    python assets/plot_vortex_ring_motion.py --format "$figure_format"
    python assets/plot_vortex_ring_energy.py --format "$figure_format"
    python assets/plot_vortex_ring_circulation.py --format "$figure_format"
done
python assets/postprocess.py
