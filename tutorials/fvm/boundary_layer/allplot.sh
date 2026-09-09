#!/bin/bash -e

python assets/plot_blasius.py --format "${1:-png}"
python assets/plot_cf.py --format "${1:-png}"
