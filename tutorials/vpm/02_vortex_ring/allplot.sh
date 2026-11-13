#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_vortex_ring_motion
python -m assets.plot_vortex_ring_energy
python -m assets.plot_vortex_ring_circulation
python -m assets.plot_vortex_ring_scenes
python -m assets.render_vortex_ring_animation
