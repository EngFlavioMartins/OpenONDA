# Finite-volume flow

The finite-volume method (FVM) solves constant-density incompressible flow on a static mesh. Use it where walls, pressure and thin shear layers must be resolved. The tutorials teach it step by step:

| To learn | Start with |
| --- | --- |
| Periodic flow, time and convection schemes | [Taylor–Green vortex](../tutorials/fvm/taylor_green/README.md) |
| Wall-resolved laminar flow, skin friction | [Flat-plate boundary layer](../tutorials/fvm/boundary_layer/README.md) |
| Inlet/outlet flow, separation and reattachment | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |
| Bluff-body shedding, force coefficients | [Square cylinder](../tutorials/fvm/cube_flow/README.md) |
| Meshing around an STL body | [STL body](../tutorials/fvm/cartesian_mesher/README.md) |
| Far-field boundaries, surface pressure | [NACA 0012 airfoil](../tutorials/fvm/airfoil_flow/README.md) |
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

## Two ways to set up a case

`FVMCase` groups a mesh, `Numerics`, boundaries, initial fields and a `RunPlan`, and runs through `FVMSolver`:

```python
import numpy as np

from openonda import fvm

x = np.linspace(0.0, 2.0, 17)  # m, 16 cells along x and y
mesh = fvm.mesher.box_mesh_3d(x, x, np.array([0.0, 0.125]), empty_spanwise=True)

case = fvm.FVMCase(
    name="uniform-flow",
    directory="first-fvm-case",
    mesh=mesh,
    numerics=fvm.Numerics(
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
        coupling=fvm.PimpleControl(algorithm="PISO", n_correctors=2),
    ),
    boundaries=(
        fvm.BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
        fvm.BoundaryConfig.outlet("outlet"),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(end_time=0.1, time_step_size=1.0e-3),
)
with fvm.FVMSolver(case) as solver:
    solver.run()
```

The tutorials use the equivalent flat form: an `FVMSetup` holding `time`, `schemes`, `linear`, `pimple`, `transport`, `boundaries`, `samplers` and `turbulence`, passed with the mesh to `fvm.create_fvm_solver(setup, case_dir=..., mesh=...)`. Both accept the same configuration objects.

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

`CartesianMesher` refines with `BoxRefinement`, `SphereRefinement`, `ConeRefinement` and `LineRefinement` regions, `PatchRefinement` on a named surface, `FeatureRefinement` along sharp edges, and `BoundaryLayers` for wall-normal prism layers. Cell sizes are $H/2^\ell$ with $H$ the background `max_cell_size`: a patch target selects the first size at or below the request, a region target a size strictly below it. `CartesianMesher.effective_cell_size(target)` previews the result. The [STL body](../tutorials/fvm/cartesian_mesher/README.md) shows a complete mesher; the [cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) shows an extruded body-fitted 2-D mesh.

A generated mesh is saved next to the case as `assets/<case_name>_mesh.npz` (with a `.vtu` copy for ParaView) and reused while its inputs are unchanged. The solver checks cell volumes and closure on construction.

Refine wall-normal spacing where viscous shear matters; the [flat plate](../tutorials/fvm/boundary_layer/README.md) uses a stretched wall-normal grid.

## Boundary conditions

Every mesh patch needs a `BoundaryConfig` with the same name.

| Constructor | Physical condition | Tutorial |
| --- | --- | --- |
| `inlet(name, [U, V, W])` | Prescribed velocity, zero normal pressure gradient | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |
| `outlet(name, kinematic_pressure=0)` | Pressure datum, zero-gradient outflow velocity | [Square cylinder](../tutorials/fvm/cube_flow/README.md) |
| `wall(name)` | Stationary no-slip wall | [Flat plate](../tutorials/fvm/boundary_layer/README.md) |
| `slip(name)` | Impermeable, shear-free boundary | [STL body](../tutorials/fvm/cartesian_mesher/README.md) |
| `freestream(name, [U, V, W])` | Far field: inflow takes the freestream velocity, outflow the pressure datum | [Airfoil](../tutorials/fvm/airfoil_flow/README.md) |
| `cyclic(name, neighbour)` | Periodic pair; declare both directions | [Taylor–Green](../tutorials/fvm/taylor_green/README.md) |
| `empty(name)` | Span faces of a one-cell 2-D mesh | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |

A time-dependent inflow uses `VelocityRamp(initial=..., final=..., start_time=..., end_time=...)`, attached to patches through `FVMSetup(velocity_boundaries=(VelocityBoundary(...),))`. With `normal_only=True` the ramp prescribes only the normal velocity, so a slip patch can carry a transient transverse inflow; the [cylinder reference](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/README.md) uses one to trigger shedding.

## Time and discretisation

| Setting | Options |
| --- | --- |
| `PimpleControl(algorithm=...)` | `PISO`: transient, `n_correctors` pressure corrections. `PIMPLE`: adds `n_outer_correctors` nonlinear corrections, with `velocity_relaxation`/`pressure_relaxation`. `SIMPLE`: steady state through `solver.solve_steady()`. Each PISO correction updates the momentum prediction with the previous correction's velocity change (Issa 1986). |
| `DiscretizationConfig(time_scheme=...)` | `euler_implicit` (first order) or `backward` (second-order BDF). |
| `convection_scheme` | `upwind` (robust, dissipative), `central` (accurate when resolved, may oscillate), bounded `limitedLinear` (default), `linearUpwind`, `LUST`, `vanLeer`, `MUSCL`, `minmod`, `superbee`. |
| `gradient_scheme` | `lsq` (default) or `gauss`. |
| `TimeConfig(adjustment=MaximumCourantTimeStep(maximum, maximum_time_step_size))` | Adapts the step to a Courant limit. |

`LinearSolverConfig` sets the momentum and pressure solver tolerances; `ComputeConfig(operator_backend="numba")` speeds up the discrete operators. `LinearSolverConfig(momentum_preconditioner=...)` selects the preconditioner of the BiCGStab momentum solve:

| Option | Preconditioner |
| --- | --- |
| `dilu` (default) | Diagonal incomplete LU, $(D^*+L)\,D^{*-1}(D^*+U)$ with the original off-diagonal coefficients, as in OpenFOAM. Rebuilt from the current matrix in one sweep; suited to the diagonally dominant transient momentum matrices. |
| `jacobi` | Diagonal scaling only. Cheapest per iteration, but needs more iterations. |
| `ilu` | SciPy's thresholded incomplete LU (`ilu_drop_tolerance`, `ilu_fill_factor`, `reuse_ilu`). Stronger for poorly conditioned steady solves; its triangular solves cost far more per application. |

The preconditioner changes only the iterations, not the converged tolerance; the PETSc backend uses its own block-Jacobi preconditioner. The default `momentum_tolerance=1e-9` converges each momentum predictor well below the discretisation error: on the cylinder reference a looser `1e-6` with `momentum_relative_tolerance=0.05` shifted the developed lift amplitude by 2.6 %, while DILU keeps the tight solve cheaper than the old loose ILU one. Repeat a case with finer cells and a smaller step before treating drag, shedding frequency or wall shear as converged; `fvm.update_grid_study` and `fvm.analyse_grid_study` estimate the discretisation error from a grid family, as in the [cube reference flow](../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/README.md).

## Turbulence and LES

The default `TurbulenceConfig.none()` sets $\nu_t=0$, which the laminar tutorials use. For large-eddy simulation, set `turbulence` to one of:

| Model | Closure and default coefficients |
| --- | --- |
| `smagorinsky()` | $\nu_t=(C_s\Delta)^2\lvert S\rvert$, $C_s=0.17$ |
| `equilibrium_smagorinsky()` | Subgrid energy in local equilibrium, $C_k=0.094$, $C_e=1.048$ |
| `dynamic_smagorinsky()` | Germano–Lilly coefficient from a test filter, globally averaged |
| `wale()` | WALE, $C_w=0.325$; vanishes near walls without damping |
| `sigma()` | Singular values of the velocity gradient, $C_\sigma=1.35$ |

Here $\lvert S\rvert=\sqrt{2S_{ij}S_{ij}}$ and $\Delta=V^{1/3}$, or $\sqrt{V/b}$ on a one-cell 2-D mesh of thickness $b$. Refining the mesh changes both the resolved scales and the modelled viscosity. WALE or sigma suit wall-bounded flows without a homogeneous direction. The [cube cases](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) use equilibrium Smagorinsky.

## Observations

Samplers write CSV histories to `samples/` on a `RunSchedule(every_time=...)` or `every_n_steps`:

| Sampler | Output |
| --- | --- |
| `ForceSampler(patch_names, reference_velocity, reference_area, reference_length)` | Pressure and viscous forces, moments and coefficients $C_D=F_x/(\tfrac12\rho U^2A)$ |
| `LineSampler(start, end, n_points=...)` | Velocity, vorticity and pressure along a line |
| `SurfaceSampler(point, normal, bounds, spacing)` | The same fields on a planar grid |
| `YPlusSampler(patch_names)` | Wall $y^+$ statistics |

`fvm.compute_kinetic_energy`, `compute_enstrophy` and `compute_continuity_error` evaluate global diagnostics from saved fields. Volume fields are written to `solution/fvm.pvd` for ParaView; see [results](solution_layout.md).

## Parallel runs

Set `FVMSetup(cores=N)`. The solver launches its own MPI workers, switches to PETSc linear algebra and partitions the mesh (cyclic cases keep a replicated mesh). Partitioning pays off on meshes of a few hundred thousand cells or more; small meshes run faster serially. A partitioned backup restarts only with the same core count.
