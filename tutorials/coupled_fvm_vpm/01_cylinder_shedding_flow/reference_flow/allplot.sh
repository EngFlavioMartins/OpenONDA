#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

python ../assets/plot_cylinder_forces.py --case-dir . --format "${1:-both}"
