#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.results restore ..
python -m openonda.tutorial_runner . postprocess_grid_study
