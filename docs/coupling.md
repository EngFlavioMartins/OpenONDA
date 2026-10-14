# FVM–VPM coupling

The coupled solver splits the flow where each method is strongest. The finite-volume solver resolves the body, its no-slip boundary layer, the pressure and the near wake. Vortex particles carry the outer wake, which may travel far downstream without a mesh. Inside an overlap region both describe the same flow: the particles supply the outer boundary condition of the FVM box, and the FVM vorticity replaces the particles near the body.

| Case | Physics |
| --- | --- |
| [Cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Laminar planar shedding at $Re=150$; one FVM cell and one particle layer across the span. Includes a parameter sensitivity study. |
| [Cube](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | Separated three-dimensional flow at $Re=1000$ with LES in both solvers. |

Each case has an FVM-only `reference_flow` that solves the same problem on a large domain. Read the [FVM](fvm.md) and [VPM](vpm.md) guides first.

## How an exchange works

Both solvers advance in steps of the exchange interval $\Delta t_\mathrm{VPM}=n\,\Delta t_\mathrm{FVM}$:

1. The particles advance one step.
2. The particles' velocity and its normal derivative at the FVM box faces become the FVM boundary condition, and the FVM advances $n$ steps.
3. In the transfer region inside the box, the FVM vorticity replaces the particle representation; outside it the particle wake is kept.
4. Steps 2–3 repeat until the boundary trace converges (`interface_iterations`, tolerances `interface_normal_tolerance` in m/s and `interface_gradient_tolerance` in 1/s). Usually one pass suffices.

Match the freestream, viscosity and turbulence model in both solvers, and define $Re=U_\infty L/\nu$ with the same body length.

## Geometry and mesh

1. **FVM box.** Mesh a box around the body with a no-slip wall and an outer patch named `numericalBoundary`. Two bounds matter, measured on the region where FVM vorticity is used unblended, which ends inside the box faces by the transfer inset, the particle-only band and the blend width:
   - Keep the coupling boundary out of the body's near field. With the box faces half a body length from the wall, the drag and shedding frequency rise by 8–10% for both the cylinder and the cube.
   - Extend the unblended region downstream past the vortex-formation length, where the separated shear layers roll up and the centreline velocity fluctuation peaks. The end of the mean reverse flow is not enough. In the [cylinder case](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/coupling_findings.md), a region ending inside the formation zone lowered the lift amplitude by 5%; beyond it the error levels off near 1%.
2. **Transfer region.** `transfer_region_bounds` sets a box in metres inside the FVM box where the FVM owns the vorticity. Keep the wall-generated vorticity resolved inside it.
3. **Resolution and exchange step.** Use equal FVM cell size and particle spacing, as both tutorials do, and refine them together. Exchange as rarely as the particle stability limit allows: the particle step must stay below $1/\max|S|$, the inverse of the largest strain rate at the wall, and the solver stops with the measured ceiling if it does not. Frequent exchanges smooth the shed vortices and lower the lift.
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

- `transfer_region_bounds` (m): the transfer faces lie inside the FVM box faces by the inset. Particles in this band are never renewed and supply the boundary condition; a deeper inset, 15–20 cells for the cylinder, is more accurate, provided the box still satisfies the rules above.
- `eta_blend_width` (m): over this inward width the FVM weight rises from zero to one; the tutorials use $6h$.
- `vpm_only_width` (m): a band just inside the transfer faces that stays purely particle-owned; the tutorials use $2h$.
- `transfer_vorticity_cutoff` (1/s): the smallest vorticity renewed inside the region.
- `transfer_amplification_cap`: limits how strongly a particle strength may be corrected when the particles cannot represent a sharp near-wall target.

Renewal corrects the difference between the FVM vorticity and the field the particles represent, so a field the particles already reproduce is left unchanged, and total strength and impulse are conserved.

## Configuration

An `fvm.FVMCase` (mesh, boundaries, inner-flow output) and a `vpm.VPMCase` (particle wake and its output) in the same directory define the flow; `coupler.CouplerConfig` defines the overlap, the interface iteration and the coupled backups. `CouplerConfig.freestream` may ramp the background velocity with a `coupler.VelocityRamp`. The cube case uses:

```python
from openonda import coupler

config = coupler.CouplerConfig(
    freestream_velocity=(1.0, 0.0, 0.0),
    coupling_patch="numericalBoundary",
    transfer_region_bounds=(-1.45, 1.45, -1.45, 1.45, -1.45, 1.45),
    eta_blend_width=6 * 0.045,
    vpm_only_width=2 * 0.045,
    interface_iterations=3,
    backup=coupler.Backup(schedule=coupler.EveryTime(1.0)),
)
with coupler.CoupledSolver(FVM_CASE, VPM_CASE, config) as solver:
    solver.run(start_from="latest")
```

`CoupledSolver` builds the FVM solver on every MPI rank and the VPM solver on the root rank. It owns one exchange clock: both solvers accept exactly the same time at every exchange, `run()` advances to the common end time, and `advance()` steps one exchange. Coupled backups `solution/coupled/coupled_{step:06d}/` hold the FVM backup, the VPM backup and the interface trace of one exchange, and replace the backups of the two solvers, so neither case takes a backup schedule. Each coupled backup also writes the FVM and particle frames of its time (`solution/fvm.pvd`, `solution/vpm.pvd`); whenever the particles write a frame on `VPMCase.output`, the FVM writes one at the same time. The run status and the coupling settings are recorded in `solution/coupler_metadata.json` and the log in `solution/coupler.log`.

The FVM step must divide the exchange step, and output intervals must fall on exchanges. Read forces and profiles together with `solution/coupler_diagnostics.jsonl`, which records the boundary-trace convergence, particle counts and timings per exchange. Before trusting a result, refine the cell and particle spacing, the exchange step and the box size; the cylinder's [sensitivity study](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) shows how.
