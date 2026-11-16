#!/bin/bash
set -e

cd "$(dirname "$0")"

# Animation of the two particle wakes with the lift of each wing, written to assets/delta_wing.gif
python -m assets.render_delta_wing_animation
