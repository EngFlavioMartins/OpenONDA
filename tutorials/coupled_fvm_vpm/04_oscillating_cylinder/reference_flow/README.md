# Heaving-cylinder FVM reference at $Re=150$

The FVM alone solves the [coupled heaving cylinder](../README.md) flow on a large body-fitted mesh that moves with the cylinder. Diameter, freestream, viscosity, span, motion and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-40,60]\times[-60,60]$ m, moving with the cylinder; one periodic cell across the 1 m span |
| Motion | $y(t)=A\sin(2\pi ft)$, $A=0.2$ m, $f=0.18$ Hz, set by `fvm.HeavingMotion` |
| Cell size | 0.04 m near the body, 0.08 m to $x=6$ m, 0.16 m to $x=12$ m, 0.64 m elsewhere |
| Boundaries | Velocity inlet and sides, zero-pressure outlet, no-slip cylinder |
| Time step; end time | 0.008 s; 50 s |

The whole mesh heaves, so the inlet, outlet and sides move with the cylinder. Their values are inertial-frame quantities: the inlet and both sides hold the undisturbed stream, which the moving boundaries pass through, and the outlet fixes the pressure. This is the unbounded stream that the coupled particles describe. Slip side walls would be wrong here: moving along their normal, they push the whole fluid column up and down with the mesh.

## Run

```bash
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest backup. The mesh is kept as `assets/reference_oscillating_cylinder_mesh.npz`. `figures/forces` shows $C_D$ and $C_L$. The case also writes transverse velocity profiles at $x=2$ and $4$ m, fixed in the inertial frame, and fields every 0.2 s from 25 s, with the wall time of each step in `solution/performance.jsonl`; `../allplot.sh` compares them with the coupled case.
