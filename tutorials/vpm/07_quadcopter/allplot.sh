#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_quadcopter_performance
python -m assets.plot_quadcopter_wake
python -m assets.plot_quadcopter_vorticity_history
python -m assets.render_quadcopter_animation
