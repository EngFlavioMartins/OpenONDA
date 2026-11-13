#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_decay --history solution/history.csv --output "figures/taylor_green_decay.png"
