#!/bin/bash -e

python assets/plot_profile.py --format "${1:-png}"
python assets/plot_comparison.py --format "${1:-png}"
