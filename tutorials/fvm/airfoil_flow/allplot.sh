#!/bin/bash -e

python assets/plot_forces.py --format "${1:-png}"
python assets/plot_surface_cp.py --format "${1:-png}"
python assets/plot_velocity.py --format "${1:-png}"
