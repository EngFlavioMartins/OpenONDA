#!/bin/bash
set -e

cd "$(dirname "$0")"

# Compare the coupled cube flow with the FVM-only reference reference_flow/.../grid_h0045_dt0.02:
# forces, errors by window, interface measures and cost, the wake profiles and the mid-plane
# fields, written to figures/. Each script waits, with a message, for data that are not there yet.
python -m assets.plot_comparison
python -m assets.plot_velocity_profiles
python -m assets.plot_fields

# The animation in assets/ is redrawn only from a finished run.
if python -c "import numpy as np, sys; t = np.genfromtxt('samples/forces_history.csv', delimiter=',', names=True)['time']; sys.exit(t[-1] < 60.0 - 1e-6)" 2>/dev/null; then
    python -m assets.render_cube_flow_animation
else
    echo "animation: waits for the coupled run to reach t = 60 s"
fi
