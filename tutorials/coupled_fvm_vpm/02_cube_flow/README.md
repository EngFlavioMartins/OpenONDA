# Coupled cube flow at $Re=1000$

The FVM resolves separation from the no-slip faces of a unit cube and its near wake; vortex particles carry the outer wake. Both solvers use equilibrium Smagorinsky LES with $C_k=0.094$ and $C_e=1.048$. Force coefficients use the frontal area, 1 m². The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Cube $D$; freestream $U_\infty$ | $[-0.5,0.5]^3$ m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; 0.001 m²/s |
| FVM box | $[-1.8,5.05]\times[-1.8,1.8]^2$ m, about 970k cells |
| Transfer region | 8.75 particle spacings inside the box faces (the coupler default) |
| VPM domain; outflow fringe | $[-4.5,24]\times[-3,3]^2$ m; $x\in[12,24]$ m |
| Cell and particle spacing | 0.045 m |
| FVM step; exchange step | 0.01 s; 0.04 s |
| End time; statistics | 60 s; $30\le t\le60$ s |

## Physics and numerics

The [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup) is generated from `assets/cube.stl` with uniform spacing; all outer FVM faces form the `numericalBoundary`, where the particles supply the normal velocity and the tangential part of its normal derivative ([mixed boundary condition](../../../docs/coupling.md#boundary-conditions)). Inside the transfer region, FVM vorticity replaces the particles at every exchange ([renewal](../../../docs/coupling.md#vorticity-transfer)), with a $6h$ blend and a $2h$ particle-only band; up to six FVM solves per exchange converge the boundary trace. Particles use RK2, Gaussian cores of radius $1.05h$, grid-based diffusion and FMM induction. See [FVM LES](../../../docs/fvm.md#turbulence-and-les) and [VPM diffusion and LES](../../../docs/vpm.md#diffusion-and-les) for the closures.

The layout follows the coupling rules found with the [cylinder](../01_cylinder_shedding_flow/assets/coupling_findings.md):

- **Box.** The box follows the rules of the [cylinder error study](../../../docs/reports/2026-10-23-coupling-error-sources.md), in scales measured on the FVM-only cube over $50\le t\le100$ s: the separated shear layers have a vorticity thickness $\delta_\omega=0.078$ m at the rear-face plane, where they lie at $|y|=0.89$ m, and the formation length (centreline peak of $u_\mathrm{rms}$) is 3.1 m from the cube centre. The region where FVM vorticity is used unblended starts the transfer inset plus the blend width (14.75 cells, 0.66 m) inside the box faces. Sideways it lies 8.2 $\delta_\omega$ from the cube faces and 3.2 $\delta_\omega$ from the displaced shear layers; downstream the transfer face lies 1.5 formation lengths behind the centre (reach $R=0.5$). The study found a side clearance of 1.2 $\delta_\omega$ too small (lift amplitude −4 points) and 5 sufficient; a reach $R\ge1$ would need a box 1.6 m longer, beyond the memory of the laptop this case was set up on, and the reach changed the cylinder error by at most 2 points for $R\ge0.35$.
- **Exchange step.** The coupling error grows with the number of exchanges per convective time, so the particle step is the longest the particle stability allows. The peak particle vorticity of the developing wake is about 21 s⁻¹ at $t=4$ s. On the wing the particle strain rate reached 0.69 of the peak vorticity and kept growing, so four FVM steps per exchange (0.04 s) keep the Lagrangian step number near 0.8 with a margin; five (0.05 s) would leave none. The FVM step, 0.01 s, is a Courant number of 0.22. Interface sweeps continue until the boundary trace changes by less than $10^{-5}$ (the coupler default), which the cylinder needed 1.8 times per exchange.
- **Outflow.** Shed vortices fade through an [outflow fringe](../../../docs/vpm.md#stabilization) instead of being deleted at the domain face.
- **Startup.** An impulsively started cube keeps a symmetric wake for a long time. A vortex of radius 0.5 m, half a side behind the rear face, swirls about an axis tilted from $z$ towards $-y$ and breaks both mirror symmetries so that the wake sheds; the reference starts from the same field. The freestream is steady from the start: the zero-pressure outlet of the reference resists a uniform transverse acceleration, so a transverse startup flow would give the two cases different startup impulses.

## Run

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest coupled backup in `solution/coupled/`. The mesh is generated once and kept as `assets/coupled_cube_flow_mesh.npz`. The FVM runs on eight MPI processes and the particles on the GPU. Expect about a day for 60 s of flow on a laptop, mostly in the FVM; `./allplot.sh` works on any completed portion once $t>30$ s.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | $C_D$, $C_S$ and $C_L$ against the reference; the mean drag, side-force and lift rms and the Strouhal number over $30\le t\le60$ s are printed |
| `figures/velocity_profiles` | Time-mean streamwise velocity and transverse velocity fluctuation along the centreline and along $y=0.75D$, $30\le t\le60$ s |
| `figures/velocity_profiles/t*` | Streamwise and transverse velocity along both lines at every sample time shared by both runs, every 0.25 s |
| `figures/velocity_fields/t*`, `figures/vorticity_fields/t*` | Streamwise velocity and vorticity $\omega_z$ in the mid-plane $z=0$: the reference, the coupled flow (the FVM inside its box, dashed, and the particles outside it) and their difference, at every time both runs saved slices, every 1 s. The turbulent wakes decorrelate, so once the wake is developed these differences measure phase rather than accuracy. `figures/auxiliary/field_differences.csv` lists the rms and maximum differences |
| `figures/mean_velocity_field`, `figures/mean_vorticity_field` | The same comparison for the time-mean fields over $30\le t\le60$ s, each run averaged over its own slices; the measure of accuracy for the developed wake. The rms and maximum differences are printed and listed in `figures/auxiliary/mean_field_differences.csv` |
| `figures/coupling_cost` | Wall time per second of flow time of each exchange by solver part, with the mean of the coupled run and of the reference FVM; the particle count |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particles; every particle frame, written with each coupled backup, has an FVM frame of the same time. `assets/paraview_state.py` loads both in ParaView |

The field figures compare mid-plane slices that both runs sample on one lattice, so they need the reference run with its `slice_z0` sampler. The transverse forces show no single clean shedding frequency over $30\le t\le60$ s, so the wakes are compared by their time means, not at equal shedding phase. Compare force statistics and mean profiles while refining the cell and particle spacing and the exchange step.
