#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python assets/plot_rotor_resolution.py --format "${1:-png}"
python assets/plot_rotor_loading.py --format "${1:-png}"
python assets/plot_rotor_relaxation.py --format "${1:-png}"
python assets/plot_rotor_health.py --cases coarse relaxed --format "${1:-png}"
python assets/plot_rotor_performance.py --case continue_12 --format "${1:-png}"
