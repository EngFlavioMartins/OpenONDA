#!/bin/bash
set -e

cd "$(dirname "$0")"

# Animation of the particle wake with the thrust and power coefficients, written to
# assets/rotor_flow.gif
python -m assets.render_rotor_wake_animation
