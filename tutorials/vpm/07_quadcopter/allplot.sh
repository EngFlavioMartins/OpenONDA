#!/bin/bash
set -e

cd "$(dirname "$0")"

# Animation of the four particle wakes with the mean thrust and power coefficients, written to
# assets/quadcopter_flow.gif
python -m assets.render_quadcopter_animation
