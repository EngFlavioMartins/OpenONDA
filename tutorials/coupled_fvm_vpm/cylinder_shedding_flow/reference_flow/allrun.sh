#!/usr/bin/env bash
# Run one preflight and the three r=1.5 cylinder grid-study resolutions.
set -euo pipefail

cd "$(dirname "$0")"

rm -rf solution samples figures
mkdir -p solution samples figures

python -u setup.py --dx 0.08333333333333333 --case-name very_coarse
python -u setup.py --dx 0.041666666666666664 --case-name coarse
python -u setup.py --dx 0.027777777777777776 --case-name medium
python -u setup.py --dx 0.018518518518518517 --case-name fine

python assets/postprocess.py
python assets/plot_grid_study.py
