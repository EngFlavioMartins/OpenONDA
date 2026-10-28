# NACA 4412 wing FVM reference at $10^\circ$ and $Re=1000$

The FVM alone solves the [coupled wing](../README.md) flow on a large body-fitted mesh. Wing, freestream, viscosity, LES closure, time step and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-10.24,30.72]\times[-12.8,12.8]\times[-12.8,12.8]$ m |
| Cell size | 0.04 m over the coupled FVM box, 0.08 m to $x=6$ m, 0.16 m to $x=14$ m, 0.64 m elsewhere |
| Boundaries | Velocity inlet, zero-pressure outlet, freestream faces at $y$, slip faces at $z$, no-slip wing |
| Time step; end time | 0.008 s; 24 s |

The coupled particles describe an unbounded stream, so the outer faces sit about ten chords and two spans from the wing. `setup.py` writes the same analytical wing surface as the coupled case to `assets/naca4412_wing.stl`. The domain uses the coupled FVM box as its finest region, so near the wing both runs solve on the same cells.

## Run

```bash
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest backup. The mesh is kept as `assets/reference_naca4412_flow_mesh.npz`. `figures/forces` shows $C_D$ and $C_L$; `../allplot.sh` compares them, the wake profiles at $x=2$ and 5 m and the slices with the coupled case. `solution/performance.jsonl` records the wall time of every step for the cost comparison.
