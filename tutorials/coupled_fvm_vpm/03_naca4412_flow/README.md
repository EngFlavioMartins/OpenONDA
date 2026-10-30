# Coupled NACA 4412 wing at $10^\circ$ and $Re=1000$

A finite NACA 4412 wing is pitched nose-up in a uniform stream. The FVM resolves the no-slip wing, tips and near wake; vortex particles carry the trailing vortices and outer wake. The [reference flow](reference_flow/README.md) is an FVM-only solution of the same problem.

| Quantity | Value |
| --- | --- |
| Chord; span | 1 m; 5 m |
| Angle of attack; freestream | $10^\circ$; $(1,0,0)$ m/s |
| Density; viscosity | 1 kg/m³; 0.001 m²/s |
| FVM domain | $[-1.6,2.72]\times[-1.28,1.24]\times[-3.6,3.6]$ m |
| Cell and particle spacing | 0.04 m, $c/25$ |
| FVM time step | CFL target 0.9; 0.008 s initial; 0.04 s controller ceiling |
| VPM interval | 0.032 s maximum; four nominal FVM substeps at the initial step |
| Output; backup intervals | 0.03–1.0 s; 0.5 s, fixed in physical time |
| Statistics window | $12\le t\le24$ s |

## Physics and numerical model

`assets/naca_surface.py` writes the analytical NACA 4412 surface used by the [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup). The wing is no-slip. Particle boundary data set the FVM boundary velocity and tangential velocity gradient; the FVM replaces particle vorticity inside the transfer region.

The particle wake uses Gaussian cores, RK2 transport, grid-based diffusion and FMM induction. A 12 m outflow fringe fades trailing vorticity smoothly. The initial starting vortex leaves the near wake before the statistics window begins.

The FVM selects a step from the maximum Courant number and recalculates it after each accepted substep. The 0.032 s VPM interval is a ceiling, initially corresponding to four 0.008 s FVM steps; the actual interval and substep count vary with the FVM CFL limit, particle strain and output deadlines. Every VPM interval equals the sum of its FVM substep sizes, so sampling and backup intervals remain fixed in physical time even when they do not divide the current step sizes.

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
