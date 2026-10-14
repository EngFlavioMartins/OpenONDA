# Cylinder FVM reference at $Re=150$

The FVM alone solves the [coupled cylinder](../README.md) flow on a large body-fitted mesh. Diameter, freestream, viscosity, span, startup and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-40,60]\times[-60,60]$ m; one periodic cell across the 1 m span |
| Cell size | 0.04 m near the body, 0.08 m to $x=6$ m, 0.16 m to $x=12$ m, 0.64 m elsewhere |
| Boundaries | Velocity inlet, zero-pressure outlet, slip side walls, no-slip cylinder |
| Time step; end time | 0.008 s; 100 s |

The coupled particles describe an unbounded stream, so the walls and the inlet sit far from the cylinder. With walls at $y=\pm10$ m and the inlet at $x=-8$ m, blockage accelerated the flow around the cylinder by about 1.6%, worth roughly 3% in mean drag.

## Run

```bash
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest backup. The mesh is kept as `assets/reference_cylinder_flow_mesh.npz`. `figures/forces` shows $C_D$ and $C_L$; `../allplot.sh` compares them with the coupled case. Forces, wake profiles at $x=2$ and $4$ m and the centreline through the recirculation region are written to `samples/`.
