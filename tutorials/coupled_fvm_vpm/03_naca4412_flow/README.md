# Coupled NACA 4412 wing at $10^\circ$ and $Re=1000$

A finite NACA 4412 wing is pitched nose-up in a uniform stream. The FVM resolves the no-slip wing, tips and near wake; vortex particles carry the trailing vortices and outer wake. The [reference flow](reference_flow/README.md) is an FVM-only solution of the same problem.

| Quantity | Value |
| --- | --- |
| Chord; span | 1 m; 5 m |
| Angle of attack; freestream | $10^\circ$; $(1,0,0)$ m/s |
| Density; viscosity | 1 kg/m³; 0.001 m²/s |
| FVM domain | $[-1.6,2.56]\times[-1.32,1.24]\times[-3.6,3.6]$ m, on the lattice of the reference mesh |
| Cell and particle spacing | 0.04 m, $c/25$; FVM cells of 0.02 m around the trailing edge and 0.01 m at its corners with the tips |
| FVM time step | CFL target 0.9 (about 0.0037 s, set by the 0.01 m trailing-edge cells); 0.0035 s initial |
| VPM interval | 0.08 or 0.04 s, 0.067 s on average, about 18 FVM substeps; shorter only where the particle CFL number would pass 0.9 |
| Output; backup intervals | forces 0.08 s, profiles and slices 0.2 s, fields 1.0 s; backups 0.48 s |
| Statistics window | $12\le t\le24$ s |

## Physics and numerical model

`assets/naca_surface.py` writes the analytical NACA 4412 surface used by the [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup). The section is cut at 92.7 % of the chord and closed by a half-circle trailing edge of 10 mm radius, so the wing ends at 0.937 c; flat caps close the tips. The mesher leaves fluid-cell centres inside a body part thinner than its wall cells (see the known limitation in [docs/fvm.md](../../../docs/fvm.md#mesh-setup)), so the 20 mm trailing edge has 0.02 m cells, and its corners with the tip caps, where a 0.02 m wall face still spans the whole trailing edge, have 0.01 m cells. The six box faces form one coupling patch, meshed with exact Cartesian cells at faces, edges and corners. The wing is no-slip. The trailing edge is thinner than the 0.04 m particle lattice, so a particle at it can lack a bounded moment-conserving grid stencil for the wall step of the grid diffusion; such particles keep their strength without their first moments (`wall_fallback_limit`), and the run stops if they carry more than 1 % of the total strength in one step. Without this the first run stopped at $t=0.16$ s. Particle boundary data set the FVM boundary velocity and tangential velocity gradient; the FVM replaces particle vorticity inside the transfer region.

The particle wake uses Gaussian cores, RK2 transport, grid-based diffusion and FMM induction. A 12 m outflow fringe fades trailing vorticity smoothly. The initial starting vortex leaves the near wake before the statistics window begins.

The FVM selects a step from the maximum Courant number and recalculates it after each accepted substep. The particle step is taken as rarely as the coupling study of `01_cylinder_shedding_flow` allows: its error decreases as the interval grows, up to a particle CFL number of 0.9, at which the coupler caps the interval. The 0.08 s ceiling and the 0.0035 s initial FVM step allow up to 23 FVM substeps per particle step, so the FVM CFL limit never shortens the interval. Forces are sampled at the 0.08 s ceiling; the 0.2 s profiles and slices, kept at the sample times of the reference flow, add the output times that make some intervals 0.04 s. The first run took a particle step every two FVM steps (0.006 s). Every VPM interval equals the sum of its FVM substep sizes, so sampling and backup intervals remain fixed in physical time even when they do not divide the current step sizes.

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
