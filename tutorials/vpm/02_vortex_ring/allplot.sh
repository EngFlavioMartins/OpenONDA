#!/bin/bash
set -e

cd "$(dirname "$0")"

# Animation of the dns_transposed ring, written to assets/vortex_ring.gif
python -m assets.render_vortex_ring_animation
