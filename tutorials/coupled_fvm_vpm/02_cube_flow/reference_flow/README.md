# Cube FVM reference at $Re=1000$

This body-fitted grid study supplies force and wake comparisons for the [coupled cube](../README.md). The unit cube is no-slip, with 1 m/s freestream, density 1 kg/m³ and viscosity 0.001 m²/s. Both cases use [equilibrium Smagorinsky LES](../../../../docs/fvm.md#turbulence-and-les), $C_k=0.094$, $C_e=1.048$.

The domain is $[-6.48,12.96]\times[-6.48,6.48]^2$ m, with a velocity inlet, zero kinematic-pressure outlet and lateral slip boundaries. In `setup.py`, [mesh refinement](../../../../docs/fvm.md#mesh-setup) uses spacing $h$ over the whole coupled FVM box, $[-1.8,4.3]\times[-1.8,1.8]^2$ m, $2h$ in the wake up to $x=12$ m, where the profiles are compared, and up to $8h$ outside. Time stepping limits the Courant number to 0.5 and the step to 0.005 s; end time is 100 s, and statistics are taken over $t\ge40$ s.

The startup is the same as in the coupled case: a transverse freestream of $(0,0.1,0.05)$ m/s enters at the inlet and crosses the lateral walls, then tapers smoothly to zero between 1 s and 2 s. Without it, the impulsively started cube keeps a symmetric, non-shedding wake for the whole run.

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

Run one grid with `python setup.py --name grid_h0045 -h 0.045`. `./allrun.sh` deletes previous results and runs the three grids from $t=0$; `./allcontinue.sh` resumes them from their latest backups. Each grid's mesh is kept as `assets/<case>_mesh.npz`.

Forces are sampled every 0.05 s and profiles every 0.25 s; fields are saved every 2 s and backups every 1 s. Outputs are under `samples/<case>/` and `solution/<case>/`. The plots under `figures/` compare mean drag, force RMS and Strouhal number; use enough developed shedding cycles before interpreting grid-convergence estimates. Run `../allplot.sh` for the coupled comparison.
