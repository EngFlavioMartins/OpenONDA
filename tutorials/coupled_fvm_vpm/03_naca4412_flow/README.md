# Coupled NACA 4412 wing at $10^\circ$ and $Re=1000$

A finite NACA 4412 wing is pitched nose-up in a uniform stream. The FVM resolves the no-slip wing, tips and near wake; vortex particles carry the trailing vortices and outer wake. The [reference flow](reference_flow/README.md) is an FVM-only solution of the same problem.

| Quantity | Value |
| --- | --- |
| Chord; span | 1 m; 5 m |
| Angle of attack; freestream | $10^\circ$; $(1,0,0)$ m/s |
| Density; viscosity | 1 kg/m³; 0.001 m²/s |
| FVM domain | $[-1.6,2.56]\times[-1.32,1.24]\times[-3.6,3.6]$ m, on the lattice of the reference mesh |
| Cell and particle spacing | 0.04 m, $c/25$; FVM cells of 0.02 m around the trailing edge and 0.01 m at its corners with the tips |
| FVM time step | CFL target 0.9; 0.008 s initial; 0.04 s controller ceiling |
| VPM interval | 0.015 s maximum, reduced by particle strain and output times |
| Output; backup intervals | forces 0.03 s, profiles and slices 0.2 s, fields 1.0 s; backups 0.48 s, the same times as the reference |
| Statistics window | $12\le t\le24$ s |

## Physics and numerical model

`assets/naca_surface.py` writes the analytical NACA 4412 surface used by the [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup). The section is cut at 92.7 % of the chord and closed by a half-circle trailing edge of 10 mm radius, so the wing ends at 0.937 c; flat caps close the tips. The mesher leaves fluid-cell centres inside a body part thinner than its wall cells (see the known limitation in [docs/fvm.md](../../../docs/fvm.md#mesh-setup)), so the 20 mm trailing edge has 0.02 m cells, and its corners with the tip caps, where a 0.02 m wall face still spans the whole trailing edge, have 0.01 m cells. The six box faces form one coupling patch, meshed with exact Cartesian cells at faces, edges and corners. The wing is no-slip. Particle boundary data set the FVM boundary velocity and tangential velocity gradient; the FVM replaces particle vorticity inside the transfer region.

The particle wake uses Gaussian cores, RK2 transport, grid-based diffusion and FMM induction. A 12 m outflow fringe fades trailing vorticity smoothly. The initial starting vortex leaves the near wake before the statistics window begins.

The FVM selects a step from the maximum Courant number and recalculates it after each accepted substep. The 0.015 s VPM interval is a ceiling; the actual interval and substep count vary with the FVM CFL limit, particle strain and output deadlines. Every VPM interval equals the sum of its FVM substep sizes, so sampling and backup intervals remain fixed in physical time even when they do not divide the current step sizes.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup. The FVM uses eight MPI processes and the particles use the GPU.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | Drag and lift against the FVM reference |
| `figures/velocity_profiles` | Mean transverse and spanwise wake profiles at $x/c=2$ and $5$ |
| `figures/fields` | Reference, coupled and difference fields on mid-span and cross-flow planes |
| `figures/coupling_cost` | Wall time by solver part and particle count |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particle states |
