# OpenONDA finite-volume solver

This package provides a static-mesh, constant-density incompressible
SIMPLE/PISO/PIMPLE solver for first-order polyhedral meshes. It is an R3
candidate for the configurations listed in `capabilities.json`; configurations
outside that matrix fail during setup or remain explicitly experimental.

## Public API

```python
from source.solvers.FVM import (
    BoundaryConfig,
    ExecutionConfig,
    FVMConfig,
    RunAcceptancePolicy,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)

config = FVMConfig(
    case_name="cube",
    time=TimeConfig.transient(dt=1e-3, duration=1.0),
    solver=SolverParams.pimple(
        n_correctors=2,
        n_outer=2,
        linear_solver="bicgstab",
        convection_scheme="limitedLinear",
    ),
    transport=TransportConfig.air(),
)
solver = Solver(config, case_dir="path/to/case")
solver.evolve()
solver.save_state("solution/restart.npz")
solver.write_run_manifest()
```

`Solver.from_case(path)` requires `system/controlDict`, `system/fvSolution`,
`system/fvSchemes`, `constant/transportProperties`, `0/U`, and `0/p`.
It maps PIMPLE/PISO/SIMPLE correctors; U and p solver methods, tolerances, and
iteration limits; and the time, gradient, and `div(phi,U)` schemes supported by
the Python backend. Malformed, missing, or unsupported input raises instead of
being replaced with defaults. Programmatic values in `FVMConfig.initial_U` and
`initial_p` take precedence when constructing `Solver` directly.
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

`ExecutionConfig.petsc_partitioned()` runs owned-plus-halo Gauss-gradient
PIMPLE with owned PETSc rows. Fields, global diagnostics, forces, checkpoints,
and VTU/PVTU output are invariant in 1/2/4-rank tests. Cyclic patches,
least-squares gradients, field-file initialization, and coupled FVM–VPM are
rejected in this mode. The measured weak-scaling support limit is four ranks on
the named 10-thread reference host.

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
