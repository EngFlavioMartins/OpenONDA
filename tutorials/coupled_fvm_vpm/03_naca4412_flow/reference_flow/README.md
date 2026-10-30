# NACA 4412 wing FVM reference at $10^\circ$ and $Re=1000$

This FVM-only case provides the reference for the [coupled wing](../README.md). The wing, freestream, viscosity, LES closure, time step and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-10.24,30.72]\times[-12.8,12.8]^2$ m |
| Mesh | 0.04 m around the wing, coarsening through the wake |
| Boundaries | Velocity inlet, zero-pressure outlet, freestream faces and slip spanwise faces |
| Time step; end time | 0.008 s; 24 s |

The coupled FVM domain is the finest refinement region, so both solvers resolve the wing and near wake on the same cells.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup. `../allplot.sh` compares force coefficients, wake profiles and sampled fields with the coupled case.
