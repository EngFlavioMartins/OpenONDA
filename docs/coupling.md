# FVM–VPM coupling

The coupled solver splits the flow where each method is strongest. The finite-volume solver resolves the body, its no-slip boundary layer, the pressure and the near wake. Vortex particles carry the outer wake, which may travel far downstream without a mesh. Inside an overlap region both describe the same flow: the particles supply the outer boundary condition of the FVM box, and the FVM vorticity replaces the particles near the body.

| Case | Physics |
| --- | --- |
| [Cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Laminar planar shedding at $Re=150$; one FVM cell and one particle layer across the span. Includes a parameter sensitivity study. |
| [Cube](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | Separated three-dimensional flow at $Re=1000$ with LES in both solvers. |
| [NACA 4412 wing](../tutorials/coupled_fvm_vpm/03_naca4412_flow/README.md) | A finite cambered wing with capped tips at $10^\circ$ and $Re=1000$, meshed body-fitted from an analytical STL, with LES in both solvers. |
| [Oscillating cylinder](../tutorials/coupled_fvm_vpm/04_oscillating_cylinder/README.md) | The cylinder heaving across the stream at $Re=150$; the FVM box moves with it. |

Each case has an FVM-only `reference_flow` that solves the same problem on a large domain. Read the [FVM](fvm.md) and [VPM](vpm.md) guides first.

## How an exchange works

Both solvers advance in steps of the exchange interval $\Delta t_\mathrm{VPM}=n\,\Delta t_\mathrm{FVM}$:

1. The particles advance one step.
2. The particles' velocity and its normal derivative at the FVM box faces become the FVM boundary condition, and the FVM advances $n$ steps.
3. In the transfer region inside the box, the FVM vorticity replaces the particle representation; outside it the particle wake is kept.
4. Steps 2–3 repeat until the boundary trace converges (at most `interface_iterations`, default 6, until the trace changes by less than `interface_normal_tolerance` in m/s and `interface_gradient_tolerance` in 1/s, default $10^{-5}$). The cylinder needed 1.8 passes per exchange on average; a single pass changed its drag by 3%.

Match the freestream, viscosity and turbulence model in both solvers, and define $Re=U_\infty L/\nu$ with the same body length.

## Geometry and mesh

1. **FVM box.** Mesh a box around the body with a no-slip wall and an outer patch named `numericalBoundary`. Measure two wake scales on an FVM-only or a coarse coupled solution: the vorticity thickness $\delta_\omega$ of the separated shear layers (about $3.2L/\sqrt{Re}$ for laminar separation) and the formation length $L_f$, from the body centre to the centreline peak of the streamwise velocity fluctuation. The bounds apply to the region where FVM vorticity is used unblended, which starts the transfer inset plus the blend width inside the box faces (the particle-only band lies inside the blend). They come from the [cylinder error study](reports/2026-10-23-coupling-error-sources.md):
   - Keep the coupling boundary out of the body's near field. With the box faces half a body length from the wall, the drag and shedding frequency rise by 8–10% for both the cylinder and the cube.
   - Keep the unblended region at least $5\,\delta_\omega$ from the body sideways. At $1.2\,\delta_\omega$ the cylinder lift amplitude fell by 4 points at both $Re=150$ and $250$.
   - Place the downstream transfer face at least one formation length beyond $L_f$, at $x\geq2L_f$. Shorter boxes changed the cylinder error by up to 2 points and can hide another error of opposite sign.
2. **Transfer region.** By default the transfer region lies `transfer_inset_spacings` (8.75) particle spacings inside every FVM box face, except slip planes. `transfer_region_bounds` sets it explicitly, in metres. Keep the wall-generated vorticity resolved inside it.
3. **Resolution and exchange step.** Use equal FVM cell size and particle spacing, and refine them together. Each exchange adds a small error: on the cylinder the coupling error grew with the number of exchanges per convective time, $N_x=L/(U_\infty\Delta t_\mathrm{VPM})$, as $N_x^{1.7}$, and did not depend on the FVM step. Exchange as rarely as the particle stability allows: the particle step must stay below $1/\max|S|$, the inverse of the largest strain rate of the particle field (higher than the FVM wall strain), and the solver stops with the measured ceiling if it does not. A Lagrangian step number $\max|S|\,\Delta t_\mathrm{VPM}\approx0.8$ was stable and accurate. Refine time by refining the FVM step within a fixed exchange step; refining both at a fixed $n$ does not converge the coupled solution.
4. **Particle domain.** Give the particles room for the wake you want to observe, and fade the wake out through an [outflow fringe](vpm.md#stabilization) at least two wake wavelengths deep. Removing particles abruptly at a face changes the circulation left in the domain once per shedding half-cycle, which shakes the flow at the body.
5. **Reference.** Compare against a flow with the same boundary conditions. Free-space particles describe an unbounded flow; walls or an inlet a few diameters from the body add blockage (at the cylinder, walls at $\pm10D$ raise the drag by about 3%).

### Free-slip span

For a spanwise-periodic or planar flow, bound the particles with `vpm.SlipSlabInduction` and give the FVM box, the particle domain and the transfer region the same span. A single particle layer between planes one spacing apart, with periodic FVM span faces, models planar flow; see [planar flow](vpm.md#planar-flow). The coupler stops before the first step if the spans disagree.

## Boundary conditions

The particles impose the normal velocity and the tangential part of its normal derivative on the FVM box, following Billuart, Duponcheel, Winckelmans and Chatelain, [*J. Comput. Phys.* 473 (2023) 111726](https://doi.org/10.1016/j.jcp.2022.111726):

$$
\mathbf{U}_\mathrm{FVM}\cdot\mathbf{n}=\mathbf{U}_\mathrm{VPM}\cdot\mathbf{n},
\qquad
(\mathbf{I}-\mathbf{n}\mathbf{n})\,\partial_n\mathbf{U}_\mathrm{FVM}=(\mathbf{I}-\mathbf{n}\mathbf{n})\,\partial_n\mathbf{U}_\mathrm{VPM}.
$$

The tangential velocity itself stays free, so the FVM can carry its own boundary-layer and wake structure to the faces. The pressure takes the flux-compatible `fixedFluxPressure` condition. `coupling_patch` names the patch (default `numericalBoundary`); keep walls and slip faces on separate patches.

## Vorticity transfer

The particles near the body are renewed from the FVM vorticity at every exchange on a uniform lattice, with grid-based diffusion and M4′ remeshing in the particle solver.

- `transfer_inset_spacings` (default 8.75), or `transfer_region_bounds` (m): the transfer faces lie inside the FVM box faces by the inset. Particles in this band are never renewed and supply the boundary condition; a deeper inset, 15–20 spacings for the cylinder, is more accurate, provided the box still satisfies the rules above.
- `eta_blend_width` (m; default six particle spacings): over this inward width the FVM weight rises from zero to one.
- `vpm_only_width` (m; default two particle spacings with the default ramp, none with a given ramp): a band just inside the transfer faces that stays purely particle-owned.
- `transfer_vorticity_cutoff` (1/s): the smallest vorticity renewed inside the region.
- `transfer_amplification_cap`: limits how strongly a particle strength may be corrected when the particles cannot represent a sharp near-wall target.

Renewal corrects the difference between the FVM vorticity and the field the particles represent, so a field the particles already reproduce is left unchanged, and total strength and impulse are conserved.

## Moving bodies

A body moving with prescribed translation carries its FVM box with it: set `FVMCase.mesh_motion` to an `fvm.MeshMotion` with a translating motion, for example `fvm.HeavingMotion`; see [moving meshes](fvm.md#moving-meshes). The coupler solves the particles in the frame translating with the box. A uniform frame translation leaves the vorticity equation unchanged, so the particles keep their strengths and only drift with the stream relative to the box, $\mathbf{U}_\infty-\mathbf{V}(t)$. Over each exchange the particle stream uses the mean box velocity, so the particles stay exactly placed relative to the moved box, and the box, transfer region, walls and particle diffusion grid remain fixed in that frame.

The FVM keeps inertial-frame velocities on its moving mesh; the boundary trace is converted to the inertial frame and the FVM velocity seen by the particles to the box frame. The particle positions, particle output and `domain_bounds` therefore refer to the box at the start of the run: add `mesh_translation` from `solution/coupler_diagnostics.jsonl` to place a particle field in the inertial frame. The particle domain moves with the body, and its outflow fringe stays downstream of the box. The particle stream $\mathbf{U}_\infty-\mathbf{V}$ varies during the run, and the renewal belt is sized for its largest value.

`CoupledSolver` rejects a rotating mesh at construction: the particle solver has no rotating frame.

## Configuration

An `fvm.FVMCase` (mesh, boundaries, inner-flow output) and a `vpm.VPMCase` (particle wake and its output) in the same directory define the flow; `coupler.CouplerConfig` defines the overlap, the interface iteration and the coupled backups. `CouplerConfig.freestream` may ramp the background velocity with a `coupler.VelocityRamp`. With the recommended interface defaults, the cube case uses:

```python
from openonda import coupler

config = coupler.CouplerConfig(
    freestream_velocity=(1.0, 0.0, 0.0),
    backup=coupler.Backup(schedule=coupler.EveryTime(1.0)),
)
with coupler.CoupledSolver(FVM_CASE, VPM_CASE, config) as solver:
    solver.run(start_from="latest")
```

`CoupledSolver` builds the FVM solver on every MPI rank and the VPM solver on the root rank. It owns one exchange clock: both solvers accept exactly the same time at every exchange, `run()` advances to the common end time, and `advance()` steps one exchange. Coupled backups `solution/coupled/coupled_{step:06d}/` hold the FVM backup, the VPM backup and the interface trace of one exchange, and replace the backups of the two solvers, so neither case takes a backup schedule. Each coupled backup also writes the FVM and particle frames of its time (`solution/fvm.pvd`, `solution/vpm.pvd`); whenever the particles write a frame on `VPMCase.output`, the FVM writes one at the same time. The run status and the coupling settings are recorded in `solution/coupler_metadata.json` and the log in `solution/coupler.log`.

The FVM step must divide the exchange step, and output intervals must fall on exchanges. Read forces and profiles together with `solution/coupler_diagnostics.jsonl`, which records the boundary-trace convergence, particle counts and timings per exchange. Before trusting a result, refine the cell and particle spacing, the exchange step and the box size; the cylinder's [sensitivity study](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) shows how.
