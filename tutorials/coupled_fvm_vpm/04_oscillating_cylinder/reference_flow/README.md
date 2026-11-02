# Elastically mounted cylinder, FVM reference

This FVM-only case provides the reference for the [coupled elastically mounted cylinder](../README.md). Geometry, freestream, viscosity, structure, span normalization and reduced velocities match the coupled case.

| Quantity | Value |
| --- | --- |
| Domain | $[-40,60]\times[-60,60]$ m, riding with the cylinder |
| Reduced velocities | $V^*=4$, 6 and 8 ($\mathrm{Re}=250\,V^*$) |
| Mesh | 0.02 m at the wall, 0.04 m near the cylinder, coarsening through the wake |
| Boundaries | Velocity inlet and sides, zero-pressure outlet and no-slip cylinder on springs |
| Time step; end time | 0.0025 s; 25 natural periods |

The inlet, outlet and side boundaries move with the mesh but retain their inertial-frame flow conditions. This represents the same unbounded stream as the particle model.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

`./allcontinue.sh` resumes the latest accepted backup of each run. `figures/histories_vstar_*` show the displacements and force coefficients; `../allplot.sh` overlays the amplitudes on the coupled results and the paper.
