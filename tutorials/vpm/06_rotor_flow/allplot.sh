#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_rotor_performance
python -m assets.plot_rotor_wake_planes
python -m assets.plot_rotor_loading_validation
python -m assets.plot_rotor_streamwise
python -m assets.render_rotor_animation --fps 30
python -m assets.render_rotor_wake_animation
