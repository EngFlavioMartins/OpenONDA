# Cylinder FVM reference at $Re=150$

This FVM-only case provides the reference for the [coupled cylinder](../README.md). Geometry, freestream, viscosity, startup field and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-40,60]\times[-60,60]$ m |
| Mesh | 0.04 m near the cylinder, coarsening through the wake |
| Boundaries | Velocity inlet, zero-pressure outlet, slip sides and no-slip cylinder |
| Time step; end time | 0.008 s; 100 s |

The large domain keeps the inlet and lateral boundaries away from the cylinder so that the FVM reference represents the same outer flow as the particles.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup. The case writes force coefficients, wake profiles and centreline samples to `samples/`; `../allplot.sh` performs the coupled comparison.
