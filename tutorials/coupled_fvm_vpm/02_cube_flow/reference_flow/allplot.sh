#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.postprocess_grid_study
