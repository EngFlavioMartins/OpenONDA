#!/bin/bash -e

python assets/plot_vortex_ring_motion.py --format png
python assets/plot_vortex_ring_energy.py --format png
python assets/plot_vortex_ring_circulation.py --format png
python assets/plot_vortex_ring_stability.py --format png
python assets/plot_vortex_ring_motion.py --format pdf
python assets/plot_vortex_ring_energy.py --format pdf
python assets/plot_vortex_ring_circulation.py --format pdf
python assets/plot_vortex_ring_stability.py --format pdf
python assets/plot_vortex_ring_scenes.py --available
