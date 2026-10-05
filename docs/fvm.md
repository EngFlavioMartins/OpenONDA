# Finite-volume flow

OpenONDA solves constant-density incompressible flow on static meshes. Start with [Taylor–Green decay](../tutorials/fvm/taylor_green/README.md) for periodic flow, [a flat-plate boundary layer](../tutorials/fvm/boundary_layer/README.md) for wall resolution, or [an STL body](../tutorials/fvm/cartesian_mesher/README.md) for mesh setup.

## Physical model and units

The velocity $\mathbf{U}$ and kinematic pressure $q=p/\rho$ satisfy

$$
\frac{\partial\mathbf{U}}{\partial t}
+\nabla\cdot(\mathbf{U}\otimes\mathbf{U})
=-\nabla q+\nabla\cdot(\nu_\mathrm{eff}\nabla\mathbf{U})+\mathbf{a},
\qquad \nabla\cdot\mathbf{U}=0.
$$

Here $\rho$ is constant density, $\nu_\mathrm{eff}=\nu+\nu_t$, and $\mathbf{a}$ is acceleration from any applied forcing. Set $\rho$ and molecular viscosity $\nu$ in `TransportConfig`. For a target Reynolds number, use $\nu=U_\mathrm{ref}L_\mathrm{ref}/Re$.

| Input or field | Unit | Meaning |
| --- | --- | --- |
| `density` | kg/m³ | $\rho$, used to convert pressure and forces to dimensional values |
| `kinematic_viscosity` | m²/s | Molecular viscosity $\nu$ |
| `velocity` | m/s | Cell-centred $\mathbf{U}$ |
| `kinematic_pressure` | m²/s² | $q$; multiply by $\rho$ for pressure in Pa |
| `eddy_viscosity` | m²/s | Subgrid viscosity $\nu_t$ |
| `vorticity` | 1/s | $\boldsymbol{\omega}=\nabla\times\mathbf{U}$ |
| `volumetric_face_flux` | m³/s | $\phi_f=\mathbf{U}_f\cdot\mathbf{S}_f$ |
| `cell_centre`, mesh coordinates | m | Spatial positions |
| `cell_volume` | m³ | Fluid volume $V$ |
| `face_area_vector` | m² | $\mathbf{S}_f$, outward from the owner cell |
| `time_step_size`, `end_time` | s | Time resolution and run duration |

Velocity and pressure arrays include boundary rows after the first `n_cells` physical cells. Use only the physical cells in volume integrals.

## Turbulence and LES

Without an explicit closure, `fvm.TurbulenceConfig.none()` gives $\nu_t=0$. This is suitable for the laminar tutorials; a turbulent calculation still needs enough spatial and temporal resolution.

Set `turbulence` in `Numerics` or `FVMSetup`:

| Configuration | Model and parameters |
| --- | --- |
| `fvm.TurbulenceConfig.smagorinsky()` | $\nu_t=(C_s\Delta)^2\lvert S\rvert$, default $C_s=0.17$ |
| `fvm.TurbulenceConfig.equilibrium_smagorinsky()` | Algebraic subgrid energy; energy coefficient $C_k=0.094$, dissipation coefficient $C_e=1.048$ |
| `fvm.TurbulenceConfig.wale()` | WALE, default $C_w=0.325$; viscosity decreases near walls |
| `fvm.TurbulenceConfig.sigma()` | Velocity-gradient singular-value model, default $C_\sigma=1.35$ |
| `fvm.TurbulenceConfig.dynamic_smagorinsky()` | Germano–Lilly coefficient from a volume-weighted test filter and global averaging |

The coefficients are dimensionless. $S$ is the symmetric velocity gradient and $\lvert S\rvert=\sqrt{2S_{ij}S_{ij}}$. The filter width is $\Delta=V^{1/3}$ in 3D or $\Delta=\sqrt{V/b}$ on a single-cell extruded 2D mesh of thickness $b$. Refining the mesh changes both resolved scales and modeled viscosity. Dynamic Smagorinsky uses global averaging; WALE or sigma is more suitable when a wall-bounded flow lacks a homogeneous direction.

See [coupled cube flow](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) for equilibrium Smagorinsky and [NACA 4412 flow](../tutorials/coupled_fvm_vpm/03_naca4412_flow/README.md) for classical Smagorinsky.

## Mesh setup

Use `openonda.fvm.mesher` to create a mesh before solving:

- `periodic_square_mesh(n)` gives $n\times n$ cells on $[0,2\pi]^2$ with one spanwise cell.
- `structured_box` and `coupling_box_mesh` create rectilinear 3D meshes.
- `CartesianMesher` combines a `BoxDomain`, closed `STLSurface` bodies, background spacing and local refinement.
- `GmshImporter` reads a Gmsh mesh; a saved native `.npz` mesh can also be passed to the solver.

In [the STL tutorial](../tutorials/fvm/cartesian_mesher/README.md), change `create_mesher()` to set the domain bounds in metres, assign outer patch names, and attach each solid surface to its wall patch. Use `BoxRefinement` for a wake region and `PatchRefinement` near a body. Refine wall-normal spacing where viscous shear matters; the [flat-plate case](../tutorials/fvm/boundary_layer/README.md) demonstrates a stretched wall-normal grid.

Cartesian spacings are $H/2^\ell$, where $H$ is `max_cell_size`. Patch targets select the first spacing at or below the target; box targets use a strict upper bound. Thus a target $H/4$ gives $H/4$ for a patch and $H/8$ for a box. `mesher.effective_cell_size(target, strict=True)` previews a box target; omit `strict` for a patch target. Check actual cell volumes after projection or layer generation.

Use one cell through the span and `empty` spanwise patches for a 2D model. Use physical boundary conditions on every outer face for a 3D model. The mesh must have positive volumes and closed cells; the solver checks mesh quality during construction.

## Boundary conditions

Every mesh patch needs a matching `BoundaryConfig` name. Values use the units above.

| Constructor | Physical condition | Tutorial |
| --- | --- | --- |
| `inlet("inlet", [U, 0, 0])` | Prescribed velocity; zero normal pressure gradient | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |
| `outlet("outlet", kinematic_pressure=0)` | Pressure datum with outflow velocity treatment | [Square cylinder](../tutorials/fvm/cube_flow/README.md) |
| `wall("body")` | Stationary no-slip wall | [Flat plate](../tutorials/fvm/boundary_layer/README.md) |
| `slip("sides")` | Impermeable boundary with zero shear | [STL body](../tutorials/fvm/cartesian_mesher/README.md) |
| `freestream("farfield", [U, 0, 0])` | External-flow boundary with prescribed freestream | [IBM cylinder](../tutorials/fvm/cylinder_ibm/README.md) |
| `cyclic("xmin", "xmax")` | Periodic pairing; also declare the reciprocal pair | [Taylor–Green vortex](../tutorials/fvm/taylor_green/README.md) |
| `empty("front")` | Spanwise boundary of a single-cell 2D extrusion | [Backward-facing step](../tutorials/fvm/step_profile/README.md) |

For an immersed body, the mesh contains no body-wall patch. Markers prescribe the body velocity and spread forcing to nearby cells; see [the circular-cylinder IBM case](../tutorials/fvm/cylinder_ibm/README.md).

## Time and discretisation

For a time-dependent velocity boundary, declare `fvm.VelocityRamp` with initial
and final vectors in m/s and transition endpoints in seconds. Bind it to patch
names through `velocity_boundaries=(fvm.VelocityBoundary(...),)`. The native
solver evaluates the smooth transition at each implicit endpoint and records
it in the checkpoint configuration. `normal_only=True` prescribes normal
velocity with zero tangential normal gradient; a slip patch returns to
impermeable slip when that prescribed normal velocity reaches zero.

`PimpleControl(algorithm="PISO")` advances transient flow with pressure corrections. `PIMPLE` adds nonlinear outer corrections; `n_outer_correctors` controls these and `n_correctors` controls pressure corrections. `SIMPLE` solves steady flow through `solve_steady()`.

`euler_implicit` is first-order in time; `backward` uses second-order BDF after startup. Upwind convection is more dissipative; central convection preserves a smooth resolved vortex but can oscillate in under-resolved flows. `limitedLinear` is the default. Gradients use `lsq` or `gauss`.

Reduce the time step when the Courant number grows. `MaximumCourantTimeStep` adapts it within `maximum_time_step_size`; immersed forcing also needs the Fourier limit described in [the IBM tutorial](../tutorials/fvm/cylinder_ibm/README.md). Repeat with finer cells and a smaller time step before treating drag, shedding frequency or wall shear as converged.

## Run a case

After [installation](installation.md), run the periodic tutorial from the repository root:

```bash
cd tutorials/fvm/taylor_green
./allrun.sh
./allplot.sh
```

`allrun.sh` clears previous results and starts from zero. `allcontinue.sh` resumes existing results. Edit physical constants and mesh sizes in `setup.py`; the case README explains them.

For a new uniform periodic case, save this as a Python script:

```python
from openonda import fvm

case = fvm.FVMCase(
    name="periodic-flow",
    directory="first-fvm-case",
    mesh=fvm.mesher.periodic_square_mesh(16),
    numerics=fvm.Numerics(
        transport=fvm.TransportConfig(
            density=1.0, kinematic_viscosity=0.01,
        ),
        coupling=fvm.PimpleControl(algorithm="PISO", n_correctors=2),
    ),
    boundaries=(
        fvm.BoundaryConfig.cyclic("xmin", "xmax"),
        fvm.BoundaryConfig.cyclic("xmax", "xmin"),
        fvm.BoundaryConfig.cyclic("ymin", "ymax"),
        fvm.BoundaryConfig.cyclic("ymax", "ymin"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
    ),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(end_time=0.1, time_step_size=1.0e-3),
)
with fvm.FVMSolver(case) as solver:
    solver.run()
```

The tutorials use `FVMSetup` with `create_fvm_solver`; the transport, boundary and turbulence configurations are the same.

## Read the results

Open `solution/fvm.pvd` for velocity, kinematic pressure and vorticity. Force samplers report dimensional forces and coefficients with the chosen reference velocity and area. For external flow, $C_D=F_x/(\tfrac12\rho U_\mathrm{ref}^2A_\mathrm{ref})$ and $St=fL_\mathrm{ref}/U_\mathrm{ref}$.

Check continuity, Courant number and the case-specific physical diagnostic: [vortex decay](../tutorials/fvm/taylor_green/README.md), [Blasius profiles and wall friction](../tutorials/fvm/boundary_layer/README.md), [reattachment](../tutorials/fvm/step_profile/README.md), [square-cylinder shedding](../tutorials/fvm/cube_flow/README.md), or [airfoil pressure](../tutorials/fvm/airfoil_flow/README.md). For stopping and resuming a run, see [continuation](continuation.md).
