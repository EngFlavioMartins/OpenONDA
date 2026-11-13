#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_forces
python -m assets.plot_surface_cp
python -m assets.plot_velocity
