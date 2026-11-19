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

# The animation in assets/ is redrawn only from a finished run.
if python -c "import numpy as np, sys; t = np.genfromtxt('samples/forces_history.csv', delimiter=',', names=True)['time']; sys.exit(int(t[-1] < 48.0 - 1e-6))" 2>/dev/null; then
    python -m assets.render_wing_flow_animation
else
    echo "animation: waits for the coupled run to reach t = 48 s"
fi
