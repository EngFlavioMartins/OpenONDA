#!/bin/bash
set -e

cd "$(dirname "$0")"

# Plot force coefficients of the FVM reference flow.
python -m assets.plot_forces
