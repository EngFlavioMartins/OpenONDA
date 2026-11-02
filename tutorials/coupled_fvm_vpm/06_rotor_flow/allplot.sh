#!/bin/bash
set -e

cd "$(dirname "$0")"

# Compare the meshed-blade rotor with the VLM rotor and render its wake.
python -m assets.plot_rotor_performance
python -m assets.plot_blade_forces
python -m assets.render_rotor_wake
