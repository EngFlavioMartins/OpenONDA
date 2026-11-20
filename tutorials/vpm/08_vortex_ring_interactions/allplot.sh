#!/bin/bash
set -e

cd "$(dirname "$0")"

# Spectra every 2 s and velocity-gradient invariants at t = 12 and 20 s, into samples/<case>/
python -m assets.turbulence_statistics --invariant-times 12 20
python -m assets.plot_histories
python -m assets.plot_spectra
# Particle scenes in figures/ and the DNS-LES animations in assets/
python -m assets.render_rings
