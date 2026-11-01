#!/bin/bash
set -e

cd "$(dirname "$0")"

# Plot force statistics and grid-convergence estimates.
python -m assets.postprocess_grid_study
