# OpenONDA finite-volume solver

This package provides a static-mesh, constant-density incompressible
SIMPLE/PISO/PIMPLE solver for first-order polyhedral meshes. It is qualified at
R3 for the configurations listed in `capabilities.json`; configurations
outside that matrix fail during setup or remain explicitly experimental.

The solver stores kinematic pressure ``p/ρ`` in m²/s² and volumetric face
flux ``U·Sf`` in m³/s. Constant density therefore cancels from the flow
evolution; it is applied when reporting dimensional pressure and viscous forces.

## Public API

```python
import openonda.fvm as fvm

setup = fvm.FVMSetup(
    case_name="cube",
    cores=1,
    output=fvm.OutputConfig(
        compression="lz4",
        precision="f32",
        asynchronous=True,
        ghost_layers=1,
    ),
    logging=fvm.LoggingConfig(mode="simple", interval_steps=1),
    time=fvm.TimeConfig.transient(time_step_size=1e-3, duration=1.0),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
    ),
    linear=fvm.LinearSolverConfig(
        momentum_solver="bicgstab",
        pressure_solver="amg",
    ),
    pimple=fvm.PimpleControl(n_correctors=2, n_outer_correctors=2),
    transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=1.5e-5),
    boundaries=[
        fvm.BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
        fvm.BoundaryConfig.outlet("outlet", 0.0),
    ],
    initial_velocity=[1.0, 0.0, 0.0],
    initial_kinematic_pressure=0.0,
)
# ``mesh`` may also be a mesh dictionary or a callable returning one.
with fvm.create_fvm_solver(setup, case_dir="path/to/case", mesh="mesh.msh") as solver:
    while solver.time < setup.time.end_time:
        solver.advance()
```

Configuration is provided entirely through `FVMSetup`. Initial velocity and
pressure values are supplied through `FVMSetup.initial_velocity` and
`FVMSetup.initial_kinematic_pressure`.
Nonzero `relTol` values are supported, and separate final-stage values may
override the relative tolerances for the final momentum and pressure solves.

Visualization precision is independent of solver precision. `OutputConfig`
accepts `precision="f16"`, `"f32"`, or `"f64"`; `f16` is half-quantized but
stored in float32 VTK arrays so ParaView remains compatible. FVM restart
checkpoints always remain lossless and use compact byte-shuffled/XOR history
encoding internally.

Low-level operators are imported through their defining packages:

```python
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.gmsh_importer import load_gmsh_mesh
from source.solvers.fvm.solve.equation_solver import solve_scalar_equation
```

Scalar solves accept backend-specific options in a `linear_options` mapping.
The chosen linear method is not silently replaced; `linear_failure_policy`
defaults to `"raise"`.

## Capability status

The serial reference has analytical hex/tet/prism/mixed-mesh convergence,
physical square-duct, cavity, periodic 3D flow, and LES-decay gates. A complete
1M-cell PIMPLE step used 3.62 GB peak RSS on the documented macOS ARM host.
Fixed-body IBM has transfer, force-balance, mesh-refinement, and body-fitted
force/wake evidence. One-rank FVM–VPM restart and conservation are verified,
but broader coupled cases remain experimental.

`FVMSetup(cores=N, ...)` is the public parallel interface.
`create_fvm_solver(...)` internally selects owned-plus-halo PIMPLE, owned PETSc
rows, rank-local fields, and VTU/PVTU output when `N > 1`. Visualization is
written as cell-centred, appended-binary VTK XML with LZ4 compression. Parallel
pieces include one marked overlap layer by default, so ParaView's
**Cell Data to Point Data** filter remains smooth across rank boundaries.
Fields, global diagnostics, forces, and checkpoints are invariant in the
collective MPI tests. Cyclic patches remain serial-only. The
same setup API is used by standalone FVM and coupled FVM–VPM cases; invoking
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
rectilinear and adaptive Cartesian meshers. Gmsh input is read through the
installed Gmsh API and accepts only first-order 3D tetrahedra (type 4),
hexahedra (5), prisms (6), and pyramids (7). Other dimensions and higher-order
cells fail before geometry assembly. Import provenance records the exact
contract and runtime/API version in run manifests.

The test commands and evidence files are listed in `tests/fvm/README.md`; the
machine-readable support contract is `capabilities.json`.
Measured optimization results and retained design decisions are recorded in
[`docs/fvm-performance-and-code-audit.md`](../../../docs/fvm-performance-and-code-audit.md).
