#!/bin/bash -e

python assets/plot_decay.py --history solution/history.csv --output "figures/taylor_green_decay.${1:-png}"
