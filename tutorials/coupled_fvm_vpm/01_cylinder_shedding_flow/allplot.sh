#!/bin/bash
set -e

cd "$(dirname "$0")"

# Grid independence of the FVM-only reference: force histories and wake scales of every grid.
python -m assets.plot_reference_forces
python -m assets.flow_scales

# The coupled baseline against its FVM-only reference on the same cells and step.
python -m assets.plot_forces
python -m assets.plot_velocity_profiles
python -m assets.plot_fields
python -m assets.plot_coupling_cost

# Sensitivity study: errors of every coupled run, the extrapolated reference and the error split.
python -m assets.plot_sensitivity
python -m assets.strength_ratio_history
python -m assets.plot_error_analysis
python -m assets.interface_sensitivity

# Render the developed wake of the baseline.
python -m assets.render_cylinder_animation
python -m assets.render_cylinder_wake
