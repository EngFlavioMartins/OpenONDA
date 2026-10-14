# Finite-volume flow

The finite-volume method (FVM) solves constant-density incompressible flow on a static or rigidly moving mesh. Use it where walls, pressure and thin shear layers must be resolved. The tutorials teach it step by step:

| To learn | Start with |
| --- | --- |
| Periodic flow, time and convection schemes | [Taylor–Green vortex](../tutorials/fvm/taylor_green/README.md) |
| Wall-resolved laminar flow, skin friction | [Flat-plate boundary layer](../tutorials/fvm/boundary_layer/README.md) |
| Inlet/outlet flow, separation and reattachment | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |
| Bluff-body shedding, force coefficients | [Square cylinder](../tutorials/fvm/cube_flow/README.md) |
| Meshing around an STL body | [STL body](../tutorials/fvm/cartesian_mesher/README.md) |
| Far-field boundaries, surface pressure | [NACA 0012 airfoil](../tutorials/fvm/airfoil_flow/README.md) |
| Moving meshes, prescribed body motion | [Oscillating cylinder](../tutorials/fvm/oscillating_cylinder/README.md) |
| LES, MPI parallel runs, grid studies | [Cube reference flow](../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/README.md) |

## Physical model and units

The velocity $\mathbf{U}$ and kinematic pressure $q=p/\rho$ satisfy

$$
\frac{\partial\mathbf{U}}{\partial t}
+\nabla\cdot(\mathbf{U}\otimes\mathbf{U})
=-\nabla q+\nabla\cdot(\nu_\mathrm{eff}\nabla\mathbf{U}),
\qquad \nabla\cdot\mathbf{U}=0,
$$

with $\nu_\mathrm{eff}=\nu+\nu_t$. Set the density $\rho$ and molecular viscosity $\nu$ in `TransportConfig`; for a target Reynolds number use $\nu=U_\mathrm{ref}L_\mathrm{ref}/Re$. All inputs are SI:

| Quantity | Unit | Meaning |
| --- | --- | --- |
| `density` | kg/m³ | $\rho$; converts kinematic pressure and forces to Pa and N |
| `kinematic_viscosity` | m²/s | Molecular viscosity $\nu$ |
| `velocity` | m/s | Cell-centred $\mathbf{U}$ |
| `kinematic_pressure` | m²/s² | $q$; multiply by $\rho$ for pressure in Pa |
| `eddy_viscosity` | m²/s | Subgrid viscosity $\nu_t$ |
| `vorticity` | 1/s | $\boldsymbol{\omega}=\nabla\times\mathbf{U}$ |
| `time_step_size`, `end_time` | s | Time resolution and run duration |
| Mesh coordinates, cell sizes | m | Geometry |

## Setting up a case

One immutable `FVMCase` holds the mesh, the `Numerics`, the boundaries, the initial fields, the `RunPlan` and the output settings; `FVMSolver(case)` constructs and runs it:

```python
import numpy as np

from openonda import fvm

x = np.linspace(0.0, 2.0, 17)  # m, 16 cells along x and y
mesh = fvm.mesher.box_mesh_3d(x, x, np.array([0.0, 0.125]), empty_spanwise=True)

case = fvm.FVMCase(
    name="uniform-flow",
    directory="first-fvm-case",
    numerics=fvm.Numerics(
        time_step_size=1.0e-3,
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
        pimple=fvm.PimpleConfig(algorithm="piso", n_correctors=2),
    ),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(end_time=0.1),
    output=fvm.Output(schedule=fvm.EverySteps(20)),
    mesh=mesh,
    boundaries=(
        fvm.BoundaryConfig.inlet("inlet", (1.0, 0.0, 0.0)),
        fvm.BoundaryConfig.outlet("outlet"),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ),
)
with fvm.FVMSolver(case) as solver:
    solver.run(start_from="latest")
```

`run()` writes the initial state, advances to `RunPlan.end_time` and writes the final state and a backup. A custom time loop calls `start_from("latest")`, then `advance()` per step; a driver that owns the clock, such as the coupler, passes the accepted time, `advance(time=...)`. Output, backups, samplers and log records take a schedule: `EverySteps(n)`, `EveryTime(interval)` or `FinalOnly()`.

## Mesh setup

`openonda.fvm.mesher` builds meshes in metres:

| Tool | Use |
| --- | --- |
| `box_mesh_3d(xs, ys, zs, hole_box=...)` | Rectilinear mesh from node coordinates, optionally with a box-shaped wall hole. `empty_spanwise=True` with one spanwise cell gives a 2-D model. |
| `stretched(...)`, `wall_refined_axis(...)` | Node sequences that grow geometrically away from a wall, for `box_mesh_3d`. |
| `coupling_box_mesh(...)` | Uniform box whose six sides form one coupling patch (FVM–VPM). |
| `CartesianMesher` | Octree mesh of a `BoxDomain` around closed `STLSurface` bodies, projected onto the surfaces. |
| `ExtrudedCartesianMesher` | Planar section of a `CartesianMesher` mesh, extruded through given span `levels`: body-fitted 2-D meshes. |
| `GmshImporter` | Converts a Gmsh mesh. |

`CartesianMesher` refines with `BoxRefinement` regions, `PatchRefinement` on a named surface and `FeatureRefinement` along sharp edges, and adds one automatic wrapper layer at walls. Cell sizes are $H/2^\ell$ with $H$ the background `max_cell_size`: a patch target selects the first size at or below the request, a box target a size strictly below it. `CartesianMesher.effective_cell_size(target)` previews the result. The [STL body](../tutorials/fvm/cartesian_mesher/README.md) shows a complete mesher; the [cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) shows an extruded body-fitted 2-D mesh.

A generated mesh is saved next to the case as `assets/<name>_mesh.npz` (with a `.vtu` copy for ParaView) and reused while its inputs are unchanged. The solver checks cell volumes and closure on construction.

Refine wall-normal spacing where viscous shear matters; the [flat plate](../tutorials/fvm/boundary_layer/README.md) uses a stretched wall-normal grid.

## Boundary conditions

Every mesh patch needs a `BoundaryConfig` with the same name.

| Constructor | Physical condition | Tutorial |
| --- | --- | --- |
| `inlet(name, [U, V, W])` | Prescribed velocity, zero normal pressure gradient | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |
| `outlet(name, kinematic_pressure=0)` | Pressure datum, zero-gradient outflow velocity | [Square cylinder](../tutorials/fvm/cube_flow/README.md) |
| `wall(name)` | Stationary no-slip wall | [Flat plate](../tutorials/fvm/boundary_layer/README.md) |
| `slip(name)` | Impermeable, shear-free boundary | [STL body](../tutorials/fvm/cartesian_mesher/README.md) |
| `freestream(name, [U, V, W])` | Far field split by the stream direction: upstream faces ($\mathbf{U}_\infty\cdot\mathbf{n}<0$) take the freestream velocity, downstream faces the pressure datum with zero-gradient outflow and the freestream on reverse flow | [Airfoil](../tutorials/fvm/airfoil_flow/README.md) |
| `cyclic(name, neighbour)` | Periodic pair; declare both directions | [Taylor–Green](../tutorials/fvm/taylor_green/README.md) |
| `empty(name)` | Span faces of a one-cell 2-D mesh | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |

A time-dependent inflow uses `VelocityRamp(initial=..., final=..., start_time=..., end_time=...)`, attached to patches through `FVMCase(velocity_boundaries=(VelocityBoundary(...),))`. With `normal_only=True` the ramp prescribes only the normal velocity, so a slip patch can carry a transient transverse inflow; the [cylinder reference](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/README.md) uses one to trigger shedding.

## Time and discretisation

| Setting | Options |
| --- | --- |
| `PimpleConfig(algorithm=...)` | `piso`: transient, `n_correctors` pressure corrections. `pimple`: adds `n_outer_correctors` nonlinear corrections, with `velocity_relaxation`/`pressure_relaxation`. `simple`: steady state through `solver.solve_steady()`. Each PISO correction updates the momentum prediction with the previous correction's velocity change (Issa 1986). |
| `DiscretizationConfig(time_scheme=...)` | `euler_implicit` (first order) or `backward` (second-order BDF). |
| `convection_scheme` | `upwind` (robust, dissipative), `central` (accurate when resolved, may oscillate), bounded `limited_linear` (default), `linear_upwind`, `lust`, `van_leer`, `muscl`, `minmod`, `superbee`. |
| `gradient_scheme` | `lsq` (default) or `gauss`. |
| `Numerics(time_step_control=MaximumCourantTimeStep(maximum, maximum_time_step_size))` | Adapts the step to a Courant limit; physical-time output events are reached exactly. |

`LinearSolverConfig` sets the momentum and pressure solver tolerances; `ComputeConfig(operator_backend="numba")` speeds up the discrete operators. `LinearSolverConfig(momentum_preconditioner=...)` selects the preconditioner of the BiCGStab momentum solve:

| Option | Preconditioner |
| --- | --- |
| `dilu` (default) | Diagonal incomplete LU, $(D^*+L)\,D^{*-1}(D^*+U)$ with the original off-diagonal coefficients, as in OpenFOAM. Rebuilt from the current matrix in one sweep; suited to the diagonally dominant transient momentum matrices. |
| `jacobi` | Diagonal scaling only. Cheapest per iteration, but needs more iterations. |
| `ilu` | SciPy's thresholded incomplete LU (`ilu_drop_tolerance`, `ilu_fill_factor`, `reuse_ilu`). Stronger for poorly conditioned steady solves; its triangular solves cost far more per application. |

The preconditioner changes only the iterations, not the converged tolerance; the PETSc backend uses its own block-Jacobi preconditioner. The default `momentum_tolerance=1e-9` converges each momentum predictor well below the discretisation error: on the cylinder reference a looser `1e-6` with `momentum_relative_tolerance=0.05` shifted the developed lift amplitude by 2.6 %, while DILU keeps the tight solve cheaper than the old loose ILU one. Repeat a case with finer cells and a smaller step before treating drag, shedding frequency or wall shear as converged; `fvm.update_grid_study` and `fvm.analyse_grid_study` estimate the discretisation error from a grid family, as in the [cube reference flow](../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/README.md).

## Moving meshes

`MeshMotion` moves the whole mesh as one rigid body with a prescribed motion. Pass it as `FVMCase(mesh_motion=...)`; a moving mesh needs the transient `piso` or `pimple` algorithm:

```python
motion = fvm.MeshMotion(
    fvm.HeavingMotion(amplitude=0.2, frequency=0.165, direction=(0, 1, 0)),
    rotation_degrees=(0, 0, 5),  # optional initial placement of the mesh
)
```

The motion classes mirror the VLM kinematics; times are in s, angles of the initial placement in degrees, angular velocities in rad/s:

| Class | Motion |
| --- | --- |
| `StaticMotion()` | None |
| `TranslatingMotion(velocity)` | Constant velocity |
| `RotatingMotion(angular_speed, axis, rotation_centre, acceleration_time)` | Constant rotation about a fixed axis, optional sin² spin-up |
| `HeavingMotion(amplitude, frequency, phase, direction)` | $A\sin(2\pi ft+\phi)$ along `direction` (default z) |
| `LinearPeriodicMotion(amplitude, frequency, direction, phase)` | The same sinusoidal translation with a required direction |
| `PitchingMotion(amplitude_degrees, frequency, phase, rotation_axis, rotation_centre)` | $\theta_0\sin(2\pi ft+\phi)$ about a fixed pivot |
| `SmoothRampMotion(final_velocity, acceleration_time, start_time)` | Acceleration from rest with a sin² velocity ramp |
| `ManeuverMotion(velocity_function, angular_velocity_function, rotation_centre)` | Velocities given as functions of time |
| `CompositeMotion(kinematics_components, rotation_centre)` | Sum of several motions about one pivot |

Closed-form motions place the mesh exactly; `ManeuverMotion` and `CompositeMotion` integrate their velocities with the interval-midpoint rule. The mesh as constructed is the configuration at the start time, and every rotation centre moves with the mesh.

The equations are solved in the arbitrary Lagrangian–Eulerian form

$$
\frac{d}{dt}\int_{V}\mathbf{U}\,dV+\oint_{S}\mathbf{U}\,(\mathbf{U}-\mathbf{U}_\mathrm{m})\cdot d\mathbf{S}
=-\oint_{S}q\,d\mathbf{S}+\oint_{S}\nu_\mathrm{eff}\nabla\mathbf{U}\cdot d\mathbf{S},
\qquad \oint_{S}\mathbf{U}\cdot d\mathbf{S}=0,
$$

with the mesh velocity $\mathbf{U}_\mathrm{m}=\mathbf{V}+\boldsymbol{\Omega}\times(\mathbf{x}-\mathbf{c})$. A rigid mesh keeps every cell volume and its swept face fluxes sum to zero in each cell, so a uniform stream stays exactly uniform. Momentum is convected by the flux relative to the moving faces; continuity uses the absolute flux. Velocity and pressure are absolute-frame (inertial) quantities, as are the inlet, outlet and freestream values. Inflow and outflow faces of `inlet_outlet` patches are selected by the relative flux, the upstream and downstream sides of a `freestream` patch by the freestream velocity relative to the moving faces, and the Courant number is relative to the moving cells.

| Patch | On a moving mesh |
| --- | --- |
| `wall` | No-slip wall moving with the mesh; a configured wall velocity is added, rotated with the mesh |
| `slip` | Impermeable plane; when it moves along its normal it takes the mesh normal velocity and pushes the fluid with it, so bound an open stream with inlet and outlet patches instead |
| `empty` | Must stay in its plane: a 2-D mesh translates in-plane and rotates about the span axis |
| `cyclic` | The periodic pair moves together |

Volume output writes the moved mesh, and a `ForceSampler` moment centre moves with the mesh. Restarts restore the mesh position. The [oscillating cylinder](../tutorials/fvm/oscillating_cylinder/README.md) shows a complete case.

A far-field `freestream` boundary keeps its pressure split fixed while the stream relative to its faces keeps its direction, so wake vortices crossing a static far field do not switch it. When the far field moves with an oscillating body the relative stream turns with the motion; in still fluid it is the mesh motion itself, and the whole far field switches between inflow and outflow conditions every half period. The switching disturbs the forces. For a moving far field, prefer a fixed `inlet` upstream and a fixed-pressure outlet downstream, as in the [oscillating cylinder](../tutorials/fvm/oscillating_cylinder/README.md), or a fixed zero velocity on a distant boundary around a body oscillating in still fluid. Without outer correctors, PISO convects with the flux of the previous step; add `n_outer_correctors` when the flow relative to the mesh changes quickly within a step. The FVM–VPM coupler follows translating meshes; see [moving bodies](coupling.md#moving-bodies).

## Turbulence and LES

The default `TurbulenceConfig.dns()` sets $\nu_t=0$, which the laminar tutorials use. For large-eddy simulation, set `Numerics(turbulence=...)` to one of:

| Model | Closure and default coefficients |
| --- | --- |
| `smagorinsky()` | $\nu_t=(C_s\Delta)^2\lvert S\rvert$, $C_s=0.17$ |
| `equilibrium_smagorinsky()` | Subgrid energy in local equilibrium, $C_k=0.094$, $C_e=1.048$ |

Here $\lvert S\rvert=\sqrt{2S_{ij}S_{ij}}$ and $\Delta=V^{1/3}$, or $\sqrt{V/b}$ on a one-cell 2-D mesh of thickness $b$. Refining the mesh changes both the resolved scales and the modelled viscosity. The [cube cases](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) use equilibrium Smagorinsky.

## Observations

Samplers in `FVMCase(samplers=Samplers((...)))` write to `samples/` on their `schedule` (default every step) and, with `initial=True` (default), at the initial state:

| Sampler | Output |
| --- | --- |
| `ForceSampler(patch_names, reference_velocity, reference_area, reference_length)` | Pressure and viscous forces, moments and coefficients $C_D=F_x/(\tfrac12\rho U^2A)$ |
| `LineSampler(start, end, n_points=...)` or `spacing=...` | Velocity, vorticity and pressure along a line, `samples/<name>.csv` |
| `SurfaceSampler(point, normal, bounds, spacing)` | The same fields on a planar grid with both bounds included, `samples/<name>_{step}.vts` |
| `YPlusSampler(patch_names)` | Wall $y^+$ statistics |

`fvm.compute_kinetic_energy`, `compute_enstrophy` and `compute_continuity_error` evaluate global diagnostics from saved fields. Volume fields are written on `Output.schedule` to `solution/fvm/fvm_{step:06d}.vtu`, listed in `solution/fvm.pvd` for ParaView; restart backups go to `solution/fvm/fvm_{step:06d}.npz` on `Backup.schedule`; see [results](solution_layout.md).

## Parallel runs

Set `FVMCase(cores=N)`. The solver launches its own MPI workers, switches to PETSc linear algebra and partitions the mesh (cyclic cases keep a replicated mesh). Partitioning pays off on meshes of a few hundred thousand cells or more; small meshes run faster serially. A partitioned backup restarts only with the same core count.
