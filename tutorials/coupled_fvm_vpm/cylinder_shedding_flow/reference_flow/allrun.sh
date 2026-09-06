#!/usr/bin/env bash
# Run one preflight and the three dyadic r=2 cylinder grid-study resolutions.
# Mesh publication is explicit and refuses to overwrite existing results.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p solution samples figures

python -u mesh.py --case very_coarse
python -u setup.py --dx 0.08333333333333333 --case-name very_coarse
python -u mesh.py --case coarse
python -u setup.py --dx 0.025 --case-name coarse
python -u mesh.py --case medium
python -u setup.py --dx 0.0125 --case-name medium
python -u mesh.py --case fine
python -u setup.py --dx 0.00625 --case-name fine

python assets/postprocess.py
python assets/plot_grid_study.py
