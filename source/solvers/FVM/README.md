# OpenONDA finite-volume solver

This package provides an incompressible SIMPLE/PIMPLE/PISO solver and scalar
finite-volume operators for OpenFOAM polyhedral meshes. The integrated solver
is still a research backend: production use requires validation for the target
mesh family, Reynolds number, discretisation, and parallel configuration.

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

Verified components are strict configuration/field parsing, static-mesh
validation, serial float64 NumPy/SciPy operators, structured convergence and
acceptance diagnostics, and BDF1/BDF2 restart equivalence.

Integrated 3D SIMPLE/PISO/PIMPLE, first-order Gmsh import, LES, IBM, FVM–VPM
coupling, and replicated PETSc collective solves remain experimental until the
analytical mesh-family and sustained-case release gates pass. Partitioned MPI
and FVM accelerator operators are not implemented.

Dynamic/ALE meshes, compressible flow, and multiphase flow are not supported.
Configuring dynamic mesh motion raises `NotImplementedError` because conservative
mesh-flux terms have not been implemented.

See `docs/plans/2026-07-fvm-3d-pimple-readiness-plan.md` for validation gates
and the test suite under `tests/fvm/` for the currently verified cases. The
machine-readable status is `capabilities.json`.
