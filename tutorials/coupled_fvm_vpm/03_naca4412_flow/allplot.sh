#!/bin/bash
set -e

cd "$(dirname "$0")"

# Compare the coupled wing with the FVM-only reference in reference_flow/: forces, errors by
# window, interface measures and cost, the wake profiles, the plane fields and the passage of the
# starting vortex, written to figures/. Each script waits, with a message, for data that are not
# there yet.
python -m assets.plot_comparison
python -m assets.plot_velocity_profiles
python -m assets.plot_fields
python -m assets.plot_starting_vortex
