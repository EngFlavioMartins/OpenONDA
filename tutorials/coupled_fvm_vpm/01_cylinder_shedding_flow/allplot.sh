#!/bin/bash

# Compare the coupled cylinder wake with the FVM reference.
python -m assets.plot_forces
python -m assets.plot_velocity_profiles
python -m assets.plot_fields
python -m assets.plot_coupling_cost

# Render the developed wake.
python -m assets.render_cylinder_animation
python -m assets.render_cylinder_wake
