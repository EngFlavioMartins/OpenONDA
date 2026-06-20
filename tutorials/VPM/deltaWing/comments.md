- [x] Modify the current case such that we have instead two delta wings plunging up and down (with adequate AoA to sustain lift), but out-of-phase ... the one behind is always "crossing" the wake of the one in front ... at least 10 times ... pitching+plunging, but no x-direction velocity ... free-stream flow speed in the -x direction ... initially at 5 spans behind one another.

  **Done.** `assets/delta_wing_setup.py` now builds two wings (front_wing,
  rear_wing) in a −x free-stream (background velocity, no body translation),
  yawed 180° to face the wind. They plunge+pitch with a **π phase offset** so the
  rear wing rises/falls through the front wing's wake. Front at x=+2.5 m, rear at
  x=0 (5 half-spans apart); run is ~3400 steps ≈ 10 plunge cycles.

- [x] Add sampling fields so that we can capture the wakes 1,5 and 10 spans downstream of the downstream delta wing.

  **Done.** Three streamwise-normal `SurfaceSampler` planes at 1/5/10 half-spans
  downstream (−x) of the rear wing (`wake_1span`, `wake_5span`, `wake_10span`).

- [x] Add to the post-processing a plot showing the forces acting on the front and back delta wings. Plot their positions in the bottom panel of the same figure.

  **Done.** `assets/allplot.py` writes `delta_wing_forces.png`: top panel = lift on
  front vs rear wing (per-surface loading CSVs); bottom panel = the two wings'
  plunge trajectories z(t) (reconstructed from `solution/motion_params.json`).

- [x] The figure about the amount of particles ... must be removed. Other figures did not plot due errors. Fix them.

  **Done.** The particle-count figure is removed. The circulation-history figure
  errored on bad mathtext (`r"$\\sum |\\Gamma|$"` → literal `\\sum`); fixed to
  `r"$\sum |\Gamma|$"`. Both remaining figures now render.
