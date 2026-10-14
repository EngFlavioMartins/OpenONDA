# Coupled cube flow at $Re=1000$

The FVM resolves separation from the no-slip faces of a unit cube and its near wake; vortex particles carry the outer wake. Both solvers use equilibrium Smagorinsky LES with $C_k=0.094$ and $C_e=1.048$. Force coefficients use the frontal area, 1 m². The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Cube $D$; freestream $U_\infty$ | $[-0.5,0.5]^3$ m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; 0.001 m²/s |
| FVM box | $[-1.8,4.3]\times[-1.8,1.8]^2$ m, about 860k cells |
| Transfer region | 8.75 cells inside the box faces |
| VPM domain; outflow fringe | $[-4.5,24]\times[-3,3]^2$ m; $x\in[12,24]$ m |
| Cell and particle spacing | 0.045 m |
| FVM step; exchange step | 0.01 s; 0.05 s |
| End time; statistics | 100 s; $t\ge40$ s |

## Physics and numerics

The [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup) is generated from `assets/cube.stl` with uniform spacing; all outer FVM faces form the `numericalBoundary`, where the particles supply the normal velocity and the tangential part of its normal derivative ([mixed boundary condition](../../../docs/coupling.md#boundary-conditions)). Inside the transfer region, FVM vorticity replaces the particles at every exchange ([renewal](../../../docs/coupling.md#vorticity-transfer)), with a $6h$ blend and a $2h$ particle-only band; up to six FVM solves per exchange converge the boundary trace. Particles use RK2, Gaussian cores of radius $1.05h$, grid-based diffusion and FMM induction. See [FVM LES](../../../docs/fvm.md#turbulence-and-les) and [VPM diffusion and LES](../../../docs/vpm.md#diffusion-and-les) for the closures.

The layout follows the coupling rules found with the [cylinder](../01_cylinder_shedding_flow/assets/coupling_findings.md):

- **Box.** The coupling boundary stays out of the body's near field, and the region where FVM vorticity is used unblended reaches about $1.05D$ from the centre sideways and $3.5D$ downstream, past the separated shear layers' roll-up. A box ending about $0.5D$ from the faces raised the cube drag by about 8%, as it raised the cylinder drag.
- **Exchange step.** Fewer exchanges are more accurate, up to the particle stability limit $\Delta t_\mathrm{VPM}<1/\max|S|$. At the cube edges the start-up strain rate reaches about 15 s⁻¹, a ceiling of 0.067 s. Five FVM steps per exchange (0.05 s) leave a 25% margin for the fluctuating strain of the turbulent wake.
- **Outflow.** Shed vortices fade through an [outflow fringe](../../../docs/vpm.md#stabilization) instead of being deleted at the domain face.
- **Startup.** An impulsively started cube keeps a symmetric wake for a long time. A vortex of radius 0.5 m, half a side behind the rear face, swirls about an axis tilted from $z$ towards $-y$ and breaks both mirror symmetries so that the wake sheds; the reference starts from the same field. The freestream is steady from the start: the zero-pressure outlet of the reference resists a uniform transverse acceleration, so a transverse startup flow would give the two cases different startup impulses.

## Run

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest coupled backup in `solution/coupled/`. The mesh is generated once and kept as `assets/coupled_cube_flow_mesh.npz`. The FVM runs on four MPI processes and the particles on the GPU. Expect several days for 100 s of flow on a laptop GPU, mostly in the FVM; `./allplot.sh` works on any completed portion once $t>40$ s.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | $C_D$, $C_S$ and $C_L$ against the reference; the mean drag, side-force and lift rms and the Strouhal number over $t\ge40$ s are printed |
| `figures/velocity_profiles` | Time-mean streamwise velocity and transverse velocity fluctuation along the centreline and along $y=0.75D$, $t\ge40$ s |
| `figures/coupled_fvm_vpm_fields_t*` | Coupled FVM and particle velocity in the mid-plane, every 2 s |
| `figures/reference_fvm_*_fields_t*` | The same plane against the reference; instantaneous, so the shed vortices' phases differ once shedding develops |
| `figures/coupling_diagnostics` | Boundary-trace convergence, particle count and wall time per exchange |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particles; every particle frame, written with each coupled backup, has an FVM frame of the same time. `assets/paraview_state.py` loads both in ParaView |

`./allplot.sh` first samples the saved FVM fields of both runs on a common lattice (`python -m assets.postprocess`), then draws the figures. Compare force statistics and mean profiles while refining the cell and particle spacing and the exchange step.
