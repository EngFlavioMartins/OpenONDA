# Finite-volume solver guide

This page is the reader-facing contract for OpenONDA's incompressible finite-volume
(FVM) solver. It describes the data layout, units, numerical choices, lifecycle, and
output behavior that are easy to miss when reading individual modules. The maintained
Python symbols are exported from [`openonda.fvm`](../source/solvers/fvm/__init__.py).

## Start here

The preferred construction boundary is `FVMCase`. It keeps mesh provenance, numerical
controls, boundary conditions, initial fields, output, restart, and run policy in one
immutable object. `FVMSolver` owns the mutable solution and the accepted clock.

```python
from openonda import fvm

mesh = fvm.mesher.periodic_square_mesh(16)
case = fvm.FVMCase(
    name="periodic-square",
    directory="first-fvm-case",
    mesh=mesh,
    numerics=fvm.Numerics(
        transport=fvm.TransportConfig.water(),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="central",
            gradient_scheme="lsq",
            time_scheme="euler_implicit",
        ),
        coupling=fvm.PimpleControl(algorithm="PISO", n_correctors=2),
    ),
    boundaries=(),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(
        start_time=0.0,
        end_time=0.1,
        time_step_size=1.0e-3,
        output_schedule=fvm.RunSchedule(every_n_steps=10),
    ),
)

with fvm.FVMSolver(case) as solver:
    solver.run()
```

`periodic_square_mesh` is a small structured test mesh. For a three-dimensional case,
use one of the rectilinear mesh helpers such as `fvm.mesher.coupling_box_mesh`, a
`CartesianMesher`, a `GmshImporter`, or a saved native mesh. The mesh must
be available before `FVMSolver` is constructed; geometry is computed once and cached.

The older `FVMSetup` plus `create_fvm_solver` path remains supported for tutorials and
the coupler. It is the low-level execution configuration, not a second numerical
model. New standalone applications should use `FVMCase`.

## Units and array contracts

OpenONDA uses SI units. The solver is constant-density and stores kinematic pressure.

| Quantity | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `vertex_position` | `(n_vertices, 3)` | m | Mesh vertex coordinates. |
| `face_centre` | `(n_faces, 3)` | m | Area-weighted face centres. |
| `face_area_vector` | `(n_faces, 3)` | m² | Oriented face area vector `Sf`. |
| `face_area` | `(n_faces,)` | m² | `||Sf||`. |
| `cell_centre` | `(n_cells, 3)` | m | Cell centres. |
| `cell_volume` | `(n_cells,)` | m³ | Positive fluid-cell volumes. |
| `velocity` | `(n_cells + n_boundary_faces, 3)` | m/s | Cell-centred velocity plus boundary ghost rows. |
| `kinematic_pressure` | `(n_cells + n_boundary_faces,)` | m²/s² | `p/ρ`, including ghost rows. |
| `volumetric_face_flux` | `(n_faces,)` | m³/s | `phi = U_f · Sf`; positive owner-to-neighbour. |
| `velocity_gradient` | `(n_cells, 3, 3)` | 1/s | `J_ij = ∂U_i/∂x_j`. |
| `vorticity` | `(n_cells, 3)` | 1/s | `curl(U)`. |
| `courant_number` | `(n_cells,)` | 1 | Local face-flux Courant number. |

`FieldState` is the synchronized public view of the three primary fields. Its arrays
are contiguous `float64` arrays. The first `n_cells` rows are physical cells; ghost
rows are implementation-owned boundary values and must not be treated as additional
fluid volume. In partitioned PETSc execution the local prefix and halo layout is
rank-dependent; use `solver.parallel.n_owned` and the topology view instead of assuming
that every local row is globally owned.

Mesh connectivity has one authoritative convention:

* `owners[f]` is the cell on the owner side of face `f`.
* `neighbours[f]` is defined for the first `n_interior_faces` and is the other cell.
* `face_area_vector[f]` points out of the owner cell. For an interior face this is
  therefore into the neighbour cell; for a boundary face it points out of the fluid.
* Boundary faces are stored in contiguous patch ranges described by `boundary` entries
  (`name`, `start_face`, `n_faces`, and `type`). Their virtual neighbour is the face
  centre, while their field value is held in the corresponding ghost row.
* Cell-face incidence may be a list of face-index arrays or a CSR pair of arrays. It is
  an indexing contract, not a geometric field.

The `MeshGeometry` facade exposes immutable, read-only views. It may share memory with
the solver's current `geo_data`; replacing a mesh or moving a mesh invalidates the
facade and requires a new solver. Dynamic/ALE meshes are rejected because conservative
mesh-flux terms are not implemented.

## Governing equations and discretisation

The implemented model is constant-density, incompressible Navier--Stokes in kinematic
form:

\[
  \frac{\partial U}{\partial t} + \nabla\cdot(U\otimes U)
  = -\nabla(p/\rho) + \nabla\cdot(\nu_\mathrm{eff}\nabla U) + S,
  \qquad \nabla\cdot U = 0.
\]

Here `density` is used when dimensional pressure or force is reported, while the
evolution uses `kinematic_viscosity` and `kinematic_pressure`. With turbulence enabled,
`nu_eff = nu + nu_t`; the turbulence model returns a non-negative cell field and the
solver exchanges its partition halos before assembly.

`DiscretizationConfig` selects the available convection, gradient, and time schemes.
`gauss` reconstructs gradients from face values; `lsq` uses a cell-centre least-squares
stencil. Convection schemes control the face velocity reconstruction and therefore the
boundedness/diffusion trade-off. `euler_implicit`/`backward` are first-order implicit
steps; `bdf2` uses the two accepted history levels and is only active after enough
history exists. The solver does not silently replace an unsupported choice.

`LinearSolverConfig` has separate momentum and pressure controls. Absolute tolerances
are residual stopping thresholds in the assembled equation units. Relative tolerances
are normalized against the initial residual; a nonzero final relative tolerance can be
used for the last correction. `linear_failure_action="raise"` preserves a failed solve
as an error; `direct_fallback` explicitly permits the configured direct fallback.
Pressure's constant nullspace is handled by the selected `pressure_nullspace_method`.

### SIMPLE, PISO, and PIMPLE

`PimpleControl.algorithm` selects the outer algorithm:

* `SIMPLE` is the steady algorithm. Call `solve_steady()`; calling `advance()` or
  `solve_pimple()` for a SIMPLE setup raises an error.
* `PISO` performs pressure-velocity correctors for one transient candidate step.
* `PIMPLE` combines inner pressure correctors with outer non-linear corrections.

`n_correctors` controls pressure corrections per outer pass, while
`n_outer_correctors` controls PIMPLE's outer passes. Non-orthogonal corrections apply
to the pressure Laplacian. Relaxation factors act on the assembled momentum and
pressure updates. Residual and continuity limits in `RunAcceptanceLimits` are checked
on the candidate state before it becomes an accepted time level.

## Boundary conditions

`BoundaryConfig` describes one named patch. Values are copied/normalized during
configuration and use SI units:

* velocity values are m/s;
* kinematic pressure values are m²/s²;
* volumetric flux values are m³/s;
* eddy viscosity values are m²/s.

The convenience constructors cover fixed-velocity inlets, fixed-pressure outlets,
freestream boundaries, no-slip walls, slip walls, empty patches, and cyclic pairs.
The low-level strings are OpenFOAM-style (`fixedValue`, `zeroGradient`, `inletOutlet`,
`freestream`, `slip`, `empty`, `cyclic`, and related strategies). A boundary type is
not merely a label: the solver reconstructs ghost velocity and scalar values and uses
the patch type during flux, gradient, pressure, and turbulence assembly.

`cyclic` requires a reciprocal `neighbour_patch` and is currently serial-only. `empty`
is for an extruded two-dimensional mesh and requires compatible topology. A missing
boundary patch or inconsistent cyclic pairing fails during construction rather than
being silently treated as a wall.

## Lifecycle and mutation rules

The accepted state is the only state written to restart files and scientific output.
The principal transitions are:

```text
accepted state --solve_pimple--> candidate state --advance_time--> accepted state
       |                               |
       +-- save_state/load_state       +-- error: candidate is not committed
```

* `solve_pimple(dt)` assembles and solves the current transient candidate without
  advancing `time` or `step`. It is intentionally repeatable for the coupling
  boundary-condition/pressure iteration, but repeated calls for one candidate must
  use the same `dt`.
* `advance_time()` rolls `velocity_old`/`velocity_older` and the corresponding flux
  history, increments the accepted step, advances physical time, writes diagnostics,
  dispatches samplers, and applies visualization/backup schedules.
* `advance()` selects a time step, calls `solve_pimple`, enforces acceptance limits, and
  commits with `advance_time`. It is the normal interactive one-step API.
* `run()` owns the finite `RunPlan`, initial/final output, terminal status, metadata,
  and cleanup. It may be called once. A failed candidate is not made restartable by
  continuing the same solver; load the last accepted backup into a new solver.
* `solve_steady()` owns the SIMPLE iteration loop and returns only after convergence,
  an iteration limit, or an error according to the configured policy.
* `set_initial_velocity` and `set_initial_state` are transactional pre-step setters.
  They rebuild boundary ghosts, face fluxes, and time-history levels. Once the first
  step is committed they raise `RuntimeError`.
* `save_state(path)` flushes output and atomically writes the complete, versioned time
  state. `load_state(path)` validates mesh/config identity unless
  `allow_config_change=True`; it also reconciles restart-aware output histories.
* `close()` flushes/finishes owned writers and logger resources. The context-manager
  form is recommended for interactive control.

Callbacks installed with `set_post_solve_state_callback` run after the linear solve,
before acceptance diagnostics. They may mutate the solver-owned cell fields and flux;
the solver then repairs halos/ghosts and recomputes dependent diagnostics. This hook is
the supported place for an external coupling projection, not a way to bypass the
accepted-state transaction.

## Mesh generation and quality

`openonda.fvm.mesher` contains the public mesh builders. Rectilinear meshes use
midpoint cells and deterministic owner/neighbour ordering. `CartesianMesher` accepts a
`BoxDomain`, one or more `STLSurface` objects, dyadic cell sizes, local refinements,
feature controls, and optional boundary-layer controls. `build()` returns the native
face-based dictionary, validates topology/geometry/quality, and records a
`GenerationReport` in `mesh_generation`.

The cfMesh octree uses spacings `H / 2**level`, where `H = max_cell_size`.
Boundary/patch requests select the first spacing at or below the target.
Box requests use a strict upper bound (including cfMesh's small floating-point
tolerance): equality triggers another level. For example, a box target `H/4`
resolves to `H/8`, while a patch target `H/4` resolves to `H/4`. Changing `H`
changes every available spacing and can move fixed targets across a level
threshold. `mesher.effective_cell_size(target, strict=True)` previews a box
target; omit `strict` for a boundary/patch target. Reports list these nominal
control sizes, before overlaps, 2:1 balancing, projection, and wrapper layers.
Report levels are additional levels relative to `H`; per-cell refinement
levels count from the root cube.

Mesh quality limits are construction gates, not post-processing hints. Non-positive
cell volume, invalid face closure, unsupported cell topology, a failed surface
conformance check, or a configured quality limit raises before a solver can use the
mesh. `save_native_mesh()` preserves the lossless native representation; `export_vtk`
and `export_openfoam` are interchange/output representations and should not be used as
the restart authority.

## Sampling, diagnostics, and output

FVM samplers receive the current accepted solver and return/serialize fields using the
same cell-centred geometry contract. `LineSampler`, `SurfaceSampler`, `ForceSampler`,
`IBMForceSampler`, and `YPlusSampler` have independent schedules through
`RunSchedule`. Force values are dimensionalized using density; velocity, pressure,
wall distance, and shear quantities retain their documented SI units.

Visualization is appended-binary cell-centred VTK XML. The canonical fields are
`velocity`, `kinematic_pressure`, `courant_number`, and `vorticity`; an active turbulence
model adds `eddy_viscosity`. `OutputConfig.precision` controls visualization precision
independently of compute precision. `f16` is quantized but stored in a ParaView-safe
float32 array. Asynchronous output is still part of the solver lifecycle: `flush_output`
surfaces writer failures before a restart or final metadata refresh is accepted.

Mesh backups and solver time steps also expose `cell_volume` (m³) and
`cell_equivalent_size = cbrt(cell_volume)` (m). Cartesian meshes additionally
provide `cell_size`, the nominal octree edge before projection/layers, and
`refinement_level`; layer indices are included when available. These geometry
arrays remain cell data when smoothing flow fields. Inspect **Cell Data** in
ParaView to compare nominal spacing with actual volume. Existing time steps
are not rewritten: their separate `mesh.vtu` already includes volume and
available nominal sizes, and ParaView's **Cell Size** filter can calculate
volume from old field-file geometry.

The default FVM artifacts are below the case directory, normally `solution/`
and `samples/`. `solution/fvm.pvd` is the ParaView entry point; it uses
portable relative paths to immutable fields and mesh files below `solution/fvm/`.
An explicit `solution_dir` or `samples_dir` overrides that default; inspect the
resolved paths on the solver when integrating with other tools.
At successful construction, the solver writes `fvm_metadata.json` in the resolved
solution directory. The same solver refreshes its generic configuration, mesh counts,
accepted clock, and lifecycle status as the run advances or finishes; tutorial input
files do not duplicate or extend that record.
Before mesh materialization begins, the startup summary prints the absolute path
to `mesher.log`. That file is stored directly in the resolved solution directory,
is flushed after every stage or refinement-pass event, and records failures before
they are propagated to the caller.

For a slow Cartesian build, compare the `seconds=` values on its `DONE` lines to
find the expensive stage. The built-in mesher uses NumPy and Numba on the CPU;
`ComputeConfig`'s Taichi backend applies to solver operators after meshing, not
to mesh generation. The mesher needs no additional pip or conda package.

## Parallel execution

`ComputeConfig` separates operator backend (`numpy`, `numba`, or `taichi`), linear
backend (`scipy` or `petsc`), parallel mode (`serial`, `petsc_replicated`, or
`petsc_partitioned`), and output mode. Serial mode requires SciPy. PETSc modes require
the optional MPI/PETSc dependencies and a matching launcher communicator.

Replicated mode keeps a complete mesh on each rank and exposes replicated output only
on rank zero. Partitioned mode stores owned cells plus halos on each rank; halo exchange
and global reductions are explicit operations on `ParallelContext`. Every rank must
participate in collective solve, output, and restart calls. Treat multi-rank modes as
experimental until the local qualification matrix has been rerun.

## Known boundaries

The FVM solver is not a compressible or multiphase solver. Dynamic mesh motion is
rejected, CUDA/Metal/Vulkan and mixed-precision FVM execution are not advertised by the
current qualification evidence, and IBM, analytical convergence, and performance
claims require the reports in [`docs/validation/fvm_qualification.md`](validation/fvm_qualification.md)
to be rerun on the target environment.
