#!/bin/bash

# Compare the coupled wing wake with the FVM reference.
python -m assets.plot_forces
python -m assets.plot_velocity_profiles
python -m assets.plot_fields
python -m assets.plot_coupling_cost
