#!/bin/bash
set -e

cd "$(dirname "$0")"

# Animation of the moving plate at 12 degrees, written to assets/flat_plate_wake.gif
python -m assets.render_flat_plate_animation
