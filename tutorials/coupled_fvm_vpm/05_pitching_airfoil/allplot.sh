#!/bin/bash
set -e

cd "$(dirname "$0")"

# Compare the coupled pitching-airfoil forces with the FVM-only reference.
python -m assets.plot_forces
