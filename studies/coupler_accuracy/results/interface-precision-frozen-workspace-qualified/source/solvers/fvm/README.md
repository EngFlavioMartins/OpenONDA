# OpenONDA finite-volume solver

This package provides a static-mesh, constant-density incompressible
SIMPLE/PISO/PIMPLE solver for first-order polyhedral meshes. Numerical and
performance support is evidence-gated; see `capabilities.json` and
[`docs/validation/fvm_qualification.md`](../../../docs/validation/fvm_qualification.md)
for the executable contract gate and the qualifications that still require
measured reports.

For the complete reader-facing contract, including units, indexing, equations,
lifecycle, sampling, output, and parallel ownership, see the
[`FVM solver guide`](../../../docs/fvm.md). This file remains the package-level
API note and capability status.

The solver stores kinematic pressure ``p/ρ`` in m²/s² and volumetric face
flux ``U·Sf`` in m³/s. Constant density therefore cancels from the flow
evolution; it is applied when reporting dimensional pressure and viscous forces.

## Public API

```python
import openonda.fvm as fvm

case = fvm.FVMCase(
    name="cube",
    directory="path/to/case",
    mesh="mesh.npz",  # or a mesh dictionary/buildable mesher
    output=fvm.OutputConfig(
        compression="lz4",
        precision="f32",
        asynchronous=True,
        ghost_layers=1,
    ),
    logging=fvm.LoggingConfig(
        mode="simple",
        schedule=fvm.RunSchedule(every_time=0.05),
    ),
    backup=fvm.BackupConfig(
        schedule=fvm.RunSchedule(every_time=0.25),
        write_at_end=True,
    ),
    run=fvm.RunPlan(
        time_step_size=1e-3,
        end_time=1.0,
        output_schedule=fvm.RunSchedule(every_n_steps=20),
        adjustment=fvm.MaximumCourantTimeStep(
            maximum=0.9,
            maximum_time_step_size=5e-3,
        ),
    ),
    numerics=fvm.Numerics(
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear",
            gradient_scheme="lsq",
        ),
        linear=fvm.LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="amg",
        ),
        coupling=fvm.PimpleControl(n_correctors=2, n_outer_correctors=2),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=1.5e-5),
    ),
    boundaries=(
        fvm.BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
        fvm.BoundaryConfig.outlet("outlet", 0.0),
    ),
    initial_conditions=fvm.InitialFields(velocity=[1.0, 0.0, 0.0]),
)
with fvm.FVMSolver(case) as solver:
    solver.run()
```

The canonical construction path is `FVMCase` followed by `FVMSolver(case)`;
the older `FVMSetup`/`create_fvm_solver` path remains available for existing
coupled/tutorial callers while migration is completed. `FVMCase` resolves its
default artifacts under `solution/` and `samples/` relative to the case root.
Initial velocity and pressure values are supplied through `InitialFields`.
`TimeConfig` and `MaximumCourantTimeStep` are immutable construction objects.
The latter uses OpenFOAM-style damped growth (at most 20% per accepted step),
immediate CFL-driven reductions, an optional maximum step, and exact final-time
clipping. The running solver owns the evolving step size; it cannot be changed
through the post-construction coupling setter.
`RunSchedule` is the common immutable cadence for visualization, logging,
sampling, and automatic restart backups. Use `every_n_steps=N` to count
accepted steps or `every_time=T` for physical seconds. With maximum-Courant
control, time-based schedules constrain the selected step just like
OpenFOAM's `adjustableRunTime`, so events land on their requested times instead
of drifting to the first step after them. Cadence decisions are derived from
the accepted step/time state, so restart continuation needs no mutable output
counter.
Nonzero `relTol` values are supported, and separate final-stage values may
override the relative tolerances for the final momentum and pressure solves.

Visualization precision is independent of solver precision. `OutputConfig`
accepts `precision="f16"`, `"f32"`, or `"f64"`; `f16` is half-quantized but
stored in float32 VTK arrays so ParaView remains compatible. FVM restart
backups always remain lossless and use compact byte-shuffled/XOR history
encoding internally.

Low-level operators are imported through their defining packages:

```python
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.gmsh_importer import load_gmsh_mesh
from source.solvers.fvm.solve.equation_solver import solve_scalar_equation
```

Scalar solves accept backend-specific options in a `linear_options` mapping.
The chosen linear method is not silently replaced; `linear_failure_action`
defaults to `"raise"`.

## Capability status

The repository contains focused contract tests for serial configuration,
restart, output, diagnostics, and coupling behavior. The broader analytical,
convergence, performance, IBM, and multi-rank claims previously associated
with this solver are not certified by the evidence present in this checkout;
they remain experimental until their reports are restored and rerun. See the
evidence map for the exact boundary.

`FVMSetup(cores=N, ...)` is the legacy parallel interface; new cases should
declare the execution controls inside `FVMCase.numerics`.
`create_fvm_solver(...)` remains available for existing callers and selects
the configured backend and output mode. Visualization is written as
cell-centred, appended-binary VTK XML. Partitioned and replicated MPI behavior
must be treated as experimental until the supported matrix has been rerun.
Cyclic patches remain serial-only. The same low-level setup API is used by
standalone FVM and coupled FVM–VPM cases; invoking
`python <case_name>_setup.py` selects the canonical environment and launches any
required worker processes internally.

Numba and Taichi CPU pass matrix, RHS, one-step, and BDF2-history parity, but
neither meets the 1.5x end-to-end acceleration gate. They are parity-only
backends, not advertised accelerators. CUDA, Metal, Vulkan, float32, and mixed
precision fail configuration until independent parity and timing evidence is
available.

Dynamic/ALE meshes, compressible flow, and multiphase flow are not supported.
Configuring dynamic mesh motion raises `NotImplementedError` because conservative
mesh-flux terms have not been implemented.

The supported mesh-input contracts are deliberately narrow. Native meshes are
plain in-memory Python dictionaries, typically produced by the bundled
rectilinear and adaptive Cartesian meshers exposed through
`openonda.fvm.mesher`. Gmsh input is read through the
installed Gmsh API and accepts only first-order 3D tetrahedra (type 4),
hexahedra (5), prisms (6), and pyramids (7). Other dimensions and higher-order
cells fail before geometry assembly. Import provenance records the exact
contract and runtime/API version in solver-owned `fvm_metadata.json`.

The maintained test command and evidence policy are listed in
`docs/validation/fvm_qualification.md`; the machine-readable support contract
is `capabilities.json`.
