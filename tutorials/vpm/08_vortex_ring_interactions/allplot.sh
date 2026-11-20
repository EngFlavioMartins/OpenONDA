#!/bin/bash
set -e

cd "$(dirname "$0")"

# Spectra every 2 s and velocity-gradient invariants at t = 12 and 20 s, into samples/<case>/
python -m assets.turbulence_statistics --invariant-times 12 20
# Energy budget per wavenumber, with each closure's sink
python -m assets.energy_transfer collision_dns --times 4 8 12 16
python -m assets.energy_transfer collision_les --times 4 8 12 16
python -m assets.energy_transfer collision_les_cusp --times 4 8 12 16
python -m assets.energy_transfer leapfrog_dns --times 5 10 20 30
python -m assets.energy_transfer leapfrog_les --times 5 10 20 30
python -m assets.energy_transfer leapfrog_les_cusp --times 5 10 20 30
python -m assets.plot_histories
python -m assets.plot_spectra
# Particle scenes in figures/ and the three-run animations in assets/
python -m assets.render_rings
