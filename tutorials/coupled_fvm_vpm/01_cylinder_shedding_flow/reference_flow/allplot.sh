#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

python -m openonda.results restore ..
python postprocess_grid_study.py
