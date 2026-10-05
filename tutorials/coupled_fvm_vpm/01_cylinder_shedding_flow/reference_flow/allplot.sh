#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

python assets/plot_forces.py --format "${1:-both}"
