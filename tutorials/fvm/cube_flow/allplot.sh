#!/bin/bash -e

python assets/plot_forces.py --format "${1:-png}"
python assets/plot_vorticity.py --format "${1:-png}"
python assets/plot_wake.py --format "${1:-png}"
