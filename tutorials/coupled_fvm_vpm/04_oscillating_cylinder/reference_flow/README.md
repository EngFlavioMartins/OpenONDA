# Heaving-cylinder FVM reference at $Re=150$

This FVM-only case provides the reference for the [coupled heaving cylinder](../README.md). Geometry, freestream, viscosity, span, motion and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-40,60]\times[-60,60]$ m, translating with the cylinder |
| Heave motion | $A=0.2$ m; $f=0.18$ Hz |
| Mesh | 0.04 m near the cylinder, coarsening through the wake |
| Boundaries | Velocity inlet and sides, zero-pressure outlet and no-slip cylinder |
| Time step; end time | 0.008 s; 50 s |

The inlet, outlet and side boundaries move with the mesh but retain their inertial-frame flow conditions. This represents the same unbounded stream as the particle model.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup. The case writes transverse profiles and fields from $t=25$ s; `../allplot.sh` compares them with the coupled case.
