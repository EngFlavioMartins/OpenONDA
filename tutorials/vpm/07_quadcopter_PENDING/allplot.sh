#!/bin/bash -e

python assets/plot_quadcopter_performance.py --format "${1:-png}"
python assets/plot_quadcopter_wake.py --format "${1:-png}"
python assets/plot_quadcopter_vorticity_history.py --format "${1:-png}"
