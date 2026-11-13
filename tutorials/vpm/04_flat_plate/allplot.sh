#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_plate_polar
python -m assets.plot_plate_staticvsmoving
python -m assets.plot_plate_spanwise
python -m assets.plot_plate_velocity
python -m assets.render_flat_plate
python -m assets.render_flat_plate_animation
