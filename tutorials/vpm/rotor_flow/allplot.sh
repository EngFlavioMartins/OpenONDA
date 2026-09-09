#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python assets/plot_rotor_performance.py --format "${1:-png}"
python assets/plot_rotor_wake_planes.py --format "${1:-png}"
python assets/plot_rotor_loading_validation.py --format "${1:-png}"
