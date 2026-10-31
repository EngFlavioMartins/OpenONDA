# FVM–VPM coupling

The coupled solver splits the flow where each method is strongest. The finite-volume solver resolves the body, its no-slip boundary layer, the pressure and the near wake inside the FVM domain $\Omega_\mathrm{FVM}$. Vortex particles carry the outer wake, which may travel far downstream without a mesh. The two meet at the interface: the outer FVM boundary $\Sigma_\mathrm{FVM}$, where the particles supply the boundary data of the FVM, and the transfer region $\Omega_\mathrm{transfer}$ inside it, where the FVM vorticity replaces the particle vorticity. The names and symbols are listed in [terminology](terminology.md).

| Case | Physics |
| --- | --- |
| [Cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Laminar planar shedding at $Re=150$; one FVM cell and one particle layer across the span. Includes a parameter sensitivity study. |
| [Cube](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | Separated three-dimensional flow at $Re=1000$ with LES in both solvers. |
| [NACA 4412 wing](../tutorials/coupled_fvm_vpm/03_naca4412_flow/README.md) | A finite cambered wing with capped tips at $10^\circ$ and $Re=1000$, meshed body-fitted from an analytical STL, with LES in both solvers. |
| [Oscillating cylinder](../tutorials/coupled_fvm_vpm/04_oscillating_cylinder/README.md) | The cylinder heaving across the stream at $Re=150$; the FVM domain moves with it. |

Each case has an FVM-only `reference_flow` that solves the same problem on a large domain. Read the [FVM](fvm.md) and [VPM](vpm.md) guides first.

## Adaptive time steps

The FVM controller sets the inner-flow step from the maximum cell Courant number. With `MaximumCourantTimeStep`, reductions are immediate, growth is damped and limited to 20% per accepted selection, and `maximum_time_step_size` supplies the hard ceiling. The controller keeps its own Courant estimate: a step shortened to reach an output time does not lower the estimate, so output times a few steps apart cannot hold the step below the Courant target. The accepted step may therefore return from a shortened step by at most a factor of 2 (variable-step BDF2 is zero-stable below $1+\sqrt{2}$). This follows OpenFOAM's `adjustTimeStep` / `maxCo` / `maxDeltaT` pattern ([time-control guide](https://www.openfoam.com/documentation/user-guide/6-solving/6.1-time-and-data-inputoutput-control), [`setDeltaT.H`](https://api.openfoam.com/2506/src_2finiteVolume_2cfdTools_2general_2include_2setDeltaT_8H_source.html)).

The VPM case's `numerics.time_step_size` is the maximum coupling interval, not a fixed particle step. At the start of each interval the coupler estimates a CFL-limited FVM step and uses it with the configured FVM/VPM step ratio to set an interval ceiling. The accepted particle strain can lower that ceiling; the estimate includes a 10% safety margin, and the VPM still checks its accepted state. Within the interval, the FVM recomputes its CFL step after every accepted FVM substep. If there are $X$ substeps, their sizes may differ, so the exact clock relation is $\Delta t_\mathrm{VPM}=\sum_{j=1}^{X}\Delta t_{\mathrm{FVM},j}$; when all FVM substeps are equal this reduces to $X\,\Delta t_\mathrm{FVM}$. If the strain limit is below the current FVM CFL ceiling, both solvers take the same shortened one-step interval.

The VPM interval ends exactly at each particle output, coupled backup, VPM log event, FVM sample, FVM frame, FVM log event and run end. An FVM output time inside an interval would otherwise force a short FVM substep to reach it, followed by many growth-limited steps; an FVM output cadence finer than the maximum coupling interval therefore also shortens the VPM interval. A step that reaches an output time is accepted at that time itself, computed from the configured decimals (`EveryTime(0.1)` gives 0.3 s, not 0.30000000000000004 s), so collections, CSV time columns and backups carry the configured times. When coupling iterations are enabled, FVM output is deferred until the interval is accepted. With multiple iterations, FVM `EverySteps` samplers and frames are not supported when more than one FVM substep may occur, because intermediate states are provisional. This is the same general pattern as OpenFOAM's adjustable time step: recalculate the CFL limit from each accepted flow state and shorten steps to land on physical-time events.

1. The particles advance over the selected VPM interval to form the boundary-data predictor.
2. The particle normal velocity and normal gradient of tangential velocity on $\Sigma_\mathrm{FVM}$ drive FVM substeps to the same interval end, with linear boundary-data interpolation from the previous accepted state.
3. Inside the transfer region the FVM vorticity replaces the particle vorticity; outside it the particle wake is kept.
4. Steps 2–3 are the coupling iterations, a fixed-point (Picard) iteration at a fixed VPM predictor. They repeat until the boundary data of the corrected particles converge: at most `max_coupling_iterations` (default 6), until the normal-velocity residual is below `normal_velocity_tolerance` (m/s) and the normal-gradient residual below `normal_gradient_tolerance` (1/s), both $10^{-5}$ by default. The first estimate adds the correction of the previous interval to the predictor data. The cylinder needed 1.8 coupling iterations per interval on average; a single iteration changed its drag by 3%.

Match the free stream, viscosity and turbulence model in both solvers, and define $Re=U_\infty L/\nu$ with the same body length.

## Geometry and mesh

1. **FVM domain.** Mesh a box around the body with a no-slip wall and an outer patch named `numericalBoundary`, declared as a coupling patch (`msh.BoxPatches.coupled("numericalBoundary")`, see [mesh setup](fvm.md#mesh-setup)). The mesher then gives the outer faces, edges and corners exact, axis-aligned cells. Measure two wake scales on an FVM-only or a coarse coupled solution: the vorticity thickness $\delta_\omega$ of the separated shear layers (about $3.2L/\sqrt{Re}$ for laminar separation) and the formation length $L_f$, from the body centre to the centreline peak of the streamwise velocity fluctuation. The rules below apply to the replacement region, where the FVM vorticity replaces the particle vorticity without blending; it starts the transfer inset plus the blend end depth $\delta_1$ inside $\Sigma_\mathrm{FVM}$. They come from the [cylinder error study](reports/2026-10-23-coupling-error-sources.md):
   - Keep the outer FVM boundary out of the near field of the body. With $\Sigma_\mathrm{FVM}$ half a body length from the wall, the drag and shedding frequency rise by 8–10% for both the cylinder and the cube.
   - Keep the replacement region at least $5\,\delta_\omega$ from the body sideways (the lateral clearance $\ell_\mathrm{side}$). At $1.2\,\delta_\omega$ the cylinder lift amplitude fell by 4 points at both $Re=150$ and $250$.
   - Place the downstream transfer surface beyond the formation point. Wake reaches $\ell_\mathrm{wake}$ of 0.35 to 2.3 formation lengths beyond $L_f$ changed the cylinder error by up to 2 points without a trend. Do not choose the reach by the size of the total error: a short FVM domain can cancel another error of opposite sign.
   - Size the FVM domain before the coupling interval: the domain error and the coupling-interval error do not add.
2. **Transfer region.** By default the transfer surface $\Sigma_\mathrm{transfer}$ lies `transfer_inset_spacings` (8.75) particle spacings inside every face of $\Sigma_\mathrm{FVM}$, except slip planes. `transfer_region_bounds` sets the transfer region explicitly, in metres. Keep the wall-generated vorticity resolved inside it.
3. **Resolution and time step.** Use equal FVM cell size and particle spacing, and refine them together. Each coupling interval adds a small coupling error: on the cylinder the error grew with the number of VPM intervals per convective time, $D/(U_\infty\Delta t_\mathrm{VPM})$, as its power 1.7. Keep the Lagrangian CFL number $\Lambda=\Delta t_\mathrm{VPM}\max|S|$ below one; the coupler limits the interval using the strain rate of the accepted particle field, and the solver still rejects a state that exceeds its configured limit. The strain rate grows as the cells are refined, so verify the time-step range at the finest planned resolution. Refine the FVM mesh and CFL target together.
4. **Particle domain.** Give the particles room for the wake you want to observe, and fade the wake out through an [outflow fringe](vpm.md#stabilization) at least two wake wavelengths deep. Removing particles abruptly at a face changes the total vortex strength left in the domain once per shedding half-cycle, which shakes the flow at the body.
5. **Reference.** Compare against a flow with the same boundary conditions. Free-space particles describe an unbounded flow; walls or an inlet a few diameters from the body add blockage (at the cylinder, walls at $\pm10D$ raise the drag by about 3%).

### Free-slip span

For a spanwise-periodic or planar flow, bound the particles with `vpm.SlipSlabInduction` and give the FVM domain, the particle domain and the transfer region the same span. A single particle layer between planes one spacing apart, with periodic FVM span faces, models planar flow; see [planar flow](vpm.md#planar-flow). The coupler stops before the first step if the spans disagree.

## Boundary conditions

The particles impose a mixed boundary condition on $\Sigma_\mathrm{FVM}$: the normal velocity $u_n$ and the normal gradient of the tangential velocity $\mathbf{g}_t$, following Billuart, Duponcheel, Winckelmans and Chatelain, [*J. Comput. Phys.* 473 (2023) 111726](https://doi.org/10.1016/j.jcp.2022.111726):

$$
\mathbf{u}^\mathrm{FVM}\cdot\mathbf{n}=\mathbf{u}^\mathrm{VPM}\cdot\mathbf{n},
\qquad
\mathbf{P}_t\,\mathbf{G}^\mathrm{FVM}\mathbf{n}=\mathbf{g}_t^\mathrm{VPM}=\mathbf{P}_t\,\mathbf{G}^\mathrm{VPM}\mathbf{n},
\qquad
\mathbf{P}_t=\mathbf{J}-\mathbf{n}\mathbf{n}^\mathsf{T}.
$$

The condition is direction-aware. Where the particle velocity enters the FVM domain, $u_n\le 0$, the face takes the full particle velocity: an inflow boundary needs every velocity component, and a tangential gradient condition there would let the FVM feed its own tangential velocity back into the inflow. Where it leaves, $u_n\ge u_\mathrm{tr}$, the mixed condition above applies and the tangential velocity stays free, so the FVM carries its own boundary-layer and wake structure out through the faces. In between, the tangential face velocity blends the two with a smooth weight of $u_n$; $u_\mathrm{tr}$ is 0.1 of the larger of the freestream speed and the RMS boundary speed. The weight depends only on the particle data, so flow may enter and leave through any face and reverse on it. A net volume flux of the particle velocity through $\Sigma_\mathrm{FVM}$ within the tolerance $(h/L_\mathrm{FVM})^2$ is removed by a uniform normal-velocity correction; a larger flux stops the run. The pressure takes the flux-consistent `fixedFluxPressure` condition. `outer_boundary_patch` names the patch (default `numericalBoundary`); keep walls and slip faces on separate patches.

The coupler compares the FVM face velocity with the particle velocity after every coupling interval. A mismatch above `boundary_mismatch_warning` (0.2) times the reference speed logs a warning with the face; above `boundary_mismatch_limit` (1.0) the run stops with `BoundaryMismatchError`, naming the face and its velocity difference. A mismatch of the order of the flow speed is not physical: it marks an unstable boundary closure or corrupted boundary data.

## Vorticity transfer

At every VPM interval the FVM vorticity replaces the particle vorticity on the transfer lattice, a uniform lattice of spacing $h$ shared with the grid-based diffusion of the particles. The particles of the remeshing region, the transfer region extended outward by $L_\mathrm{remesh}$, are remeshed onto the lattice with the M4′ kernel. The FVM target strength of a node is the Stokes integral of the reconstructed FVM velocity over the faces of its lattice cube. With $d$ the depth below the transfer surface, the blending weight $\eta$ is zero in the zero-weight layer $d\le\delta_0$, rises with a cosine and reaches one at $d=\delta_1$; deeper lies the replacement region.

- `transfer_inset_spacings` (default 8.75), or `transfer_region_bounds` (m): the transfer surface lies the inset $L_\mathrm{inset}$ inside $\Sigma_\mathrm{FVM}$. Particles in this inset band never receive FVM vorticity; they supply the boundary data. On the cylinder a deeper inset reduced the lift-amplitude coupling error slightly (−1.67 %, −1.42 %, −1.29 % at 4, 8.75 and 14 spacings in the error study); the default stays at 8.75, and a deeper inset needs a correspondingly larger FVM domain so that the rules above still hold.
- `blend_start_depth` $\delta_0$ (m; default two particle spacings): the zero-weight layer below the transfer surface keeps its particle vorticity.
- `blend_end_depth` $\delta_1$ (m; default six particle spacings): the blending weight reaches one; zero disables blending.
- `transfer_removal_threshold` $\tau_\mathrm{transfer}$ (1/s): the removal threshold of the replacement region. It falls linearly with the blending weight to the GBD removal threshold $\tau_\mathrm{VPM}$ at the transfer surface.
- `max_strength_ratio` $\kappa_\mathrm{max}$: limits the strength correction where the particles cannot represent a sharp near-wall target, through the strength ratio $\kappa$ of the largest limited strength to the peak target strength.
- `max_moment_correction_fraction`: the largest moment correction, as a fraction of the transferred strength.

The strength correction acts on the weighted mismatch $\Gamma_\mathrm{mis}=\eta\,(\Gamma^\mathrm{FVM}-\mathcal M_\sigma[\Gamma^\mathrm{VPM}])$ between the FVM target and the particle-represented vorticity, so a field the particles already reproduce is left unchanged. Weak strengths are then removed by smooth thresholding, and the moment correction restores the total vortex strength and linear impulse.

## Moving bodies

A body moving with prescribed translation carries its FVM domain with it: set `FVMCase.mesh_motion` to an `fvm.MeshMotion` with a translating motion, for example `fvm.HeavingMotion`; see [moving meshes](fvm.md#moving-meshes). The coupler solves the particles in the frame translating with the FVM domain. A uniform frame translation leaves the vorticity equation unchanged, so the particles keep their strengths and only drift with the stream relative to the mesh, $\mathbf{U}_\infty-\mathbf{V}(t)$. Over each shared step the particle stream uses the mean mesh velocity, so the particles stay exactly placed relative to the moved mesh, and the FVM domain, transfer region, walls and particle diffusion grid remain fixed in that frame.

The FVM keeps inertial-frame velocities on its moving mesh; the boundary data are converted to the inertial frame and the FVM velocity seen by the particles to the mesh frame. The particle positions, particle output and `domain_bounds` therefore refer to the FVM domain at the start of the run: add `mesh_translation` from `solution/coupler_diagnostics.jsonl` to place a particle field in the inertial frame. The particle domain moves with the body, and its outflow fringe stays downstream of the FVM domain. The particle stream $\mathbf{U}_\infty-\mathbf{V}$ varies during the run, and the remeshing extension $L_\mathrm{remesh}$ is sized for its largest value.

`CoupledSolver` rejects a rotating mesh at construction: the particle solver has no rotating frame.

## Configuration

An `fvm.FVMCase` (mesh, boundaries, inner-flow output) and a `vpm.VPMCase` (particle wake and its output) in the same directory define the flow; `coupler.CouplerConfig` defines the vorticity transfer, the coupling iterations and the coupled backups. `CouplerConfig.freestream` may ramp the free-stream velocity with a `coupler.VelocityRamp`. The FVM case selects its CFL target and maximum step; the VPM case sets the maximum coupling interval. With the recommended interface defaults, the cube case uses:

```python
from openonda import coupler

config = coupler.CouplerConfig(
    freestream_velocity=(1.0, 0.0, 0.0),
    backup=coupler.Backup(schedule=coupler.EveryTime(1.0)),
)
with coupler.CoupledSolver(FVM_CASE, VPM_CASE, config) as solver:
    solver.run(start_from="latest")
```

`CoupledSolver` builds the FVM solver on every MPI rank and the VPM solver on the root rank. `run()` advances both solvers to their common end time; each `advance()` accepts one VPM interval containing an integer number of FVM substeps. In adaptive coupled runs, `EveryTime` schedules for samples, frames and coupled backups are exact physical-time deadlines, while `EverySteps` counts accepted steps of the solver that owns that schedule. Coupled backups `solution/coupled/coupled_{step:06d}/` hold both solver states and the boundary data; they replace the backups of the two solvers, so neither case takes a separate backup schedule. Version 16 fixed-step, version 17 shared-step and version 18 separate-clock backups remain readable. Each coupled backup also writes the FVM and particle frames of its time (`solution/fvm.pvd`, `solution/vpm.pvd`); whenever the particles write a frame on `VPMCase.output`, the FVM writes one at the same time. The run status and the coupling settings are recorded in `solution/coupler_metadata.json` and the log in `solution/coupler.log`.

Read forces and profiles together with `solution/coupler_diagnostics.jsonl`, which records per VPM interval the boundary-data residuals of coupling iterations, vorticity-transfer conservation, particle counts and timings. Before trusting a result, refine the cell and particle spacing, CFL target and maximum step, and the size of the FVM domain; the cylinder's [sensitivity study](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) shows how.
