#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_delta_wing_forces
python -m assets.plot_delta_wing_force_cycles
python -m assets.plot_delta_wing_circulation_history
python -m assets.plot_delta_wing_wake_streamwise
python -m assets.plot_delta_wing_wake_vertical
python -m assets.render_delta_wing_animation
