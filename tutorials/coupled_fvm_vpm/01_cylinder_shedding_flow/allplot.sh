#!/bin/bash
set -e
cd "$(dirname "$0")"
python -m assets.plot_forces "$@"
python -m assets.plot_velocity_profiles "$@"
python -m assets.plot_fields "$@"
python -m assets.plot_coupling_cost "$@"
python -m assets.render_cylinder_animation
python -m assets.render_cylinder_wake
python -m assets.plot_sensitivity "$@"
python -m assets.flow_scales
python -m assets.plot_error_analysis
