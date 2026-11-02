#!/bin/bash
set -e

cd "$(dirname "$0")"

# Plot the cylinder motion and force coefficients of the FVM reference flows.
python -m assets.plot_histories
