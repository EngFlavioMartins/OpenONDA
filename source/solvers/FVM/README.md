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
from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    OutputSetup,
    PimpleControl,
    SchemesConfig,
    TimeConfig,
    TransportConfig,
    setup_fvm_solver,
)

setup = FVMSetup(
    case_name="cube",
    cores=1,
    output=OutputSetup(
        compression="lz4",
        asynchronous=True,
        ghost_layers=1,
    ),
    time=TimeConfig.transient(dt=1e-3, duration=1.0),
    schemes=SchemesConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="lsq",
    ),
    linear=LinearSolverConfig(
        momentum_solver="bicgstab",
        pressure_solver="amg",
    ),
    pimple=PimpleControl(n_correctors=2, n_outer_correctors=2),
    transport=TransportConfig(density=1.0, nu=1.5e-5),
    boundaries=[
        BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
        BoundaryConfig.outlet("outlet", 0.0),
    ],
)
# ``mesh`` may also be a mesh dictionary or a callable returning one.
solver = setup_fvm_solver(setup, case_dir="path/to/case", mesh="mesh.msh")
solver.evolve()
solver.save_state("solution/restart.npz")
solver.write_run_manifest()
```

`Solver.from_case(path)` requires `system/controlDict`, `system/fvSolution`,
`system/fvSchemes`, `constant/transportProperties`, `0/U`, and `0/p`.
It maps PIMPLE/PISO/SIMPLE correctors; U and p solver methods, tolerances, and
iteration limits; and the time, gradient, and `div(phi,U)` schemes supported by
the Python backend. Malformed, missing, or unsupported input raises instead of
being replaced with defaults. Programmatic values in `FVMSetup.initial_U` and
`initial_p` take precedence when constructing the solver.
Separate `UFinal`/`pFinal` solver blocks and nonzero OpenFOAM `relTol` values
are rejected because the Python driver does not implement those stopping stages.

Low-level operators are imported through their defining packages:

```python
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.mesh.mesh_io import load_poly_mesh
from source.solvers.FVM.solve.equation_solver import solve_scalar_equation
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
`setup_fvm_solver(...)` internally selects owned-plus-halo PIMPLE, owned PETSc
rows, rank-local fields, and VTU/PVTU output when `N > 1`. Visualization is
written as cell-centred, appended-binary VTK XML with LZ4 compression. Parallel
pieces include one marked overlap layer by default, so ParaView's
**Cell Data to Point Data** filter remains smooth across rank boundaries.
Fields, global
diagnostics, forces, and checkpoints are invariant in the collective MPI
tests. Cyclic patches and field-file initialization remain serial-only. The
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

The supported mesh-input contract is deliberately narrow. OpenFOAM `polyMesh`
input must use an ASCII `FoamFile` header with format version `2.0`; binary
files, preprocessing directives, macros, and headerless files are rejected.
Gmsh input is read through the installed Gmsh API and accepts only first-order
3D tetrahedra (type 4), hexahedra (5), prisms (6), and pyramids (7). Other
dimensions and higher-order cells fail before geometry assembly. Import
provenance records the exact contract and runtime/API version in run manifests.

The test commands and evidence files are listed in `tests/fvm/README.md`; the
machine-readable support contract is `capabilities.json`.
Measured optimization results and retained design decisions are recorded in
[`docs/fvm-performance-and-code-audit.md`](../../../docs/fvm-performance-and-code-audit.md).
