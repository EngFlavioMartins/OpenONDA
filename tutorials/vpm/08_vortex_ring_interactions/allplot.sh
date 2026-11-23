#!/bin/bash
set -e

cd "$(dirname "$0")"

# The energy of the Smagorinsky runs in free space, as the others sampled it
python -m assets.free_space_energy
# Spectra every 2 s and at the stop, velocity-gradient invariants at t = 6 and 12 s
python -m assets.turbulence_statistics --invariant-times 6 12
# Energy budget per wavenumber, with each closure's sink; times a run does not reach are left out
for model in dns les les_cusp; do
  python -m assets.energy_transfer collision_$model --times 2 4 5.5 7
  python -m assets.energy_transfer leapfrog_$model --times 4 8 11.5 12 14
done
python -m assets.plot_histories
python -m assets.plot_spectra
# The numbers of the README
python -m assets.summary
# Particle scenes in figures/ and the three-run animations in assets/
python -m assets.render_rings
