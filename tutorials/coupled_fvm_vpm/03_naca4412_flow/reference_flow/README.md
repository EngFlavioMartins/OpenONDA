# NACA 4412 wing FVM reference at $10^\circ$ and $Re=1000$

This FVM-only case provides the reference for the [coupled wing](../README.md). The wing, freestream, viscosity, LES closure, time step and force normalization match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-10.24,30.72]\times[-12.8,12.8]^2$ m |
| Mesh | 0.04 m around the wing, 0.02 m around the trailing edge and 0.01 m at its corners with the tips, coarsening through the wake |
| Boundaries | Velocity inlet, zero-pressure outlet, freestream faces and slip spanwise faces |
| Time step; end time | adaptive, Courant number 0.9 (0.008 s initial, 0.04 s ceiling), as the coupled FVM; 24 s |

Inside the coupled FVM domain the cells are those of the coupled case: both meshes lie on one lattice with the same refinement boxes, and the box holds the same 1,245,536 cell centres in both. They coincide except in the outermost layer at the coupling faces, within 0.5 mm, and in 82 wall cells, within 7 mm, so both solvers resolve the wing and near wake on the same cells.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup. `../allplot.sh` compares force coefficients, wake profiles and sampled fields with the coupled case.
