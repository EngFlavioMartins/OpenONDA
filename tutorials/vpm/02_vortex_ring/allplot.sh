#!/bin/bash -e
# Usage: ./allplot.sh [png|pdf] (default: png)
cd -- "$(dirname -- "$0")"

python assets/plot_vortex_ring_motion.py --format "${1:-png}"
python assets/plot_vortex_ring_energy.py --format "${1:-png}"
python assets/plot_vortex_ring_circulation.py --format "${1:-png}"
python assets/plot_vortex_ring_stability.py --format "${1:-png}"
python assets/plot_vortex_ring_scenes.py --available --format "${1:-png}"
