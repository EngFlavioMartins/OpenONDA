# Cube FVM reference at $Re=1000$

This body-fitted grid study supplies force and wake comparisons for the [coupled cube](../README.md). The unit cube is no-slip, with 1 m/s freestream, density 1 kg/m³ and viscosity 0.001 m²/s. Both cases use [equilibrium Smagorinsky LES](../../../../docs/fvm.md#turbulence-and-les), $C_k=0.094$, $C_e=1.048$.

The domain is $[-6.48,12.96]\times[-6.48,6.48]^2$ m, with a velocity inlet, zero kinematic-pressure outlet and lateral slip boundaries. In `setup.py`, [mesh refinement](../../../../docs/fvm.md#mesh-setup) uses spacing $h$ over the whole coupled FVM box, $[-1.8,4.3]\times[-1.8,1.8]^2$ m, $2h$ in the wake up to $x=12$ m, where the profiles are compared, and up to $8h$ outside. Time stepping limits the Courant number to 0.5 and the step to 0.005 s; end time is 100 s, and statistics are taken over $t\ge40$ s.

The startup is the same as in the coupled case: a vortex behind the cube breaks both mirror symmetries of the wake. Without it, the impulsively started cube keeps a symmetric, non-shedding wake for the whole run.

```bash
./allrun.sh
./allplot.sh
```

The launcher runs these grids in sequence:

| Case | Wall spacing $h$ (m) |
| --- | ---: |
| `grid_h010125` | 0.10125 |
| `grid_h00675` | 0.0675 |
| `grid_h0045` | 0.045 |

Run one grid with `python setup.py --name grid_h0045 -h 0.045`. `python setup.py --name grid_h0045_dt001 -h 0.045 --time-step 0.01` runs the finest grid at the fixed FVM step of the coupled case, which the coupling error needs; it is not part of the grid study. `./allrun.sh` deletes previous results and runs the three grids from $t=0$; `./allcontinue.sh` resumes them from their latest backups. Each grid's mesh is kept as `assets/<case>_mesh.npz`.

Forces are sampled every 0.05 s, profiles every 0.25 s and the mid-plane $z=0$ every 1 s, on the lattice of the coupled particle slice; fields are saved every 2 s and backups every 1 s. The wall time of each step is written to `solution/<case>/performance.jsonl` for the coupled cost comparison. Outputs are under `samples/<case>/` and `solution/<case>/`. The plots under `figures/` compare mean drag, force RMS and Strouhal number; use enough developed shedding cycles before interpreting grid-convergence estimates. Run `../allplot.sh` for the coupled comparison.
