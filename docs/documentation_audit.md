# FVM/VPM documentation audit

This document records the documentation contract and the audit trail for the FVM,
VPM, and FVM--VPM coupling APIs. It is intentionally kept in the repository so a
future numerical change can update the prose and validation evidence together.

## Scope

The audit covered the public namespaces and the meaningful internal boundaries that
carry numerical state:

* FVM case/configuration, mesh generation/import, topology and geometry, cell-centred
  fields, boundary strategies, gradient/divergence/interpolation/reconstruction,
  pressure--velocity algorithms, linear solvers, time control, parallel context,
  sampling, diagnostics, output, restart, and immersed-body hooks.
* VPM case/configuration, particle distributions and analytical initializers, the
  device particle container, radial kernels, direct/tree/FMM induction, stretching,
  Runge--Kutta staging, diffusion, LES, stabilization, VLM/panel hooks, sampling,
  diagnostics, backups, solver metadata, and lifecycle orchestration.
* Coupling setup/validation, face traces, FVM interpolation, vorticity transfer,
  common-lattice and renewal paths, flux handoff, boundary history, diagnostics,
  output, and coupled restart.

The source docstrings are the hover/API contract for individual symbols. The guides
[`fvm.md`](fvm.md), [`vpm.md`](vpm.md), and [`coupling.md`](coupling.md) are the
cross-module contract for layout, units, equations, ownership, lifecycle, and known
limitations. They deliberately link to source instead of duplicating every private
kernel implementation.

## Documentation standard

New and revised public docstrings use NumPy-style sections where a symbol has a
non-trivial contract:

```text
Short purpose.

Parameters
----------
name : type
    Shape, units, default, ownership, and mutation behavior.

Returns
-------
type
    Shape, units, and whether the result aliases or copies state.

Raises
------
...

Notes
-----
Mathematical convention, numerical assumption, side effects, and lifecycle rules.

Examples
--------
...
```

Dataclass configuration objects use an `Attributes` section when documenting every
field in a constructor-level block is clearer than repeating field comments. Short
pure helpers retain concise docstrings; private Taichi kernels are documented only
when their data layout or side effects are non-obvious.

## Canonical terminology

The following terms are now used consistently across the guides and source-facing
contracts:

| Preferred term | Definition |
| --- | --- |
| particle strength (circulation vector) `Γ` | VPM vector with units m³/s, `Γ = ωV`; use scalar circulation only for a prescribed line/loop integral. |
| vorticity `ω` | Field with units 1/s; for a particle, `ω = Γ/V`. |
| vortex particle | A quadrature point carrying position, `Γ`, radius, volume, and state. |
| node / vertex | A mesh point is a vertex; a lattice point is a node. |
| cell centre | The representative position of a finite-volume cell. |
| time step / `dt` | One accepted physical interval in seconds; RK stages are temporary. |
| core radius / `sigma` | The regularization length, in metres. |
| accepted state | A state eligible for diagnostics, output, transfer, or restart. |
| candidate/stage state | A temporary or uncommitted state that must not be serialized as accepted. |

FVM uses `kinematic_pressure = p/rho` (m²/s²), `volumetric_face_flux = U·Sf`
(m³/s), owner/neighbour face orientation, and cell-centred fields. VPM uses fixed
capacity with active prefix `[0:N)`, vector particle strength `Gamma` rather than
an ambiguous scalar
"particle circulation", and explicit `sigma/h` core resolution.

## Baseline inventory and checks

The initial repository-wide `interrogate` pass over `source/solvers/fvm`,
`source/solvers/vpm`, and `source/coupler` found 3,247 definitions, 915 missing
docstrings, and 71.8% documented definitions. The audit then prioritized exported
objects, constructors, lifecycle methods, numerical operators, array-boundary helpers,
and the internal APIs used by coupling and restart. Missing documentation on trivial
generated accessors was not filled mechanically when the containing class contract
already specifies the field.

Recommended local checks after a documentation or API change:

```bash
interrogate -v -f 0 -M -m openonda source/solvers/fvm source/solvers/vpm source/coupler
ruff check source/solvers/fvm source/solvers/vpm source/coupler
python -m compileall -q source openonda
git diff --check
pytest -q
```

The package has no mandatory Sphinx build in `pyproject.toml`; Markdown is the
maintained user documentation format. Examples in the guides are written against the
public `openonda` namespace and should be smoke-tested with the target optional
dependencies. GPU/MPI examples require the corresponding runtime and are not
portable doctests.

## Representative APIs inspected

The following representative hover surfaces were checked against their current
signatures and runtime behavior:

* FVM: `FVMCase`, `InitialFields`, `FVMSetup`, `BoundaryConfig`, `TimeConfig`,
  `LinearSolverConfig`, `PimpleControl`, `FieldState`, `MeshGeometry`,
  `ParallelContext`, `CartesianMesher`, `FVMSolver.solve_pimple`,
  `FVMSolver.advance_time`, `FVMSolver.save_state`, `FVMSolver.load_state`, and
  `FVMSolver.write_vtk`.
* VPM: `VPMCase`, `Numerics`, `RunPlan`, `ParticleDistribution`,
  `VortexParticleSet`, `VortexFilament`, `VortexRing`, `Particles`,
  `DirectInduction`, `TreecodeInduction`, `FMMInduction`, `RadialVortexKernel`,
  `RungeKutta`, `EverySteps`, `EveryTime`, `FinalOnly`, `SurfaceSampler`,
  `LineSampler`, and `VPMSolver.run`/`advance`/`load_backup`.
* Coupling: `CouplerSetup`, `FVMVPMCoupler.initialize`/`run`/`solve`,
  `VorticityTransfer`, `TransferResult`, and the renewal/lattice contracts.

## Known ambiguities retained explicitly

Some compatibility behavior is intentionally documented rather than silently changed:

1. VPM `Backup` currently defaults to `solution/`; sampler output is always below
   `samples/`. These paths are part of existing restart/tutorial compatibility.
2. `SurfaceSampler.save_vtp()` historically writes a structured-grid `.vts` file.
   The method name remains for compatibility and the guide calls out the actual format.
3. VPM `DIRECT`, `TRANSPOSED`, and `MIXED` are strength-rate formulations, not
   induction backends. Treecode/FMM use a hierarchical gradient; direct induction
   uses pairwise evaluation. FMM arbitrary-target queries currently use an explicit
   shared-kernel fallback while particle RK stages remain FMM-evaluated.
4. FVM partitioned arrays are local owned-plus-halo fields, whereas serial/replicated
   arrays expose the complete mesh. Consumers must use the parallel context instead of
   assuming a global row count.

These are not numerical changes. They are documented because changing them would
alter file compatibility, restart identity, or the established public API.

## Final verification (2026-09-08)

The final checks for this overhaul produced the following evidence:

* `interrogate -v -f 0 -M -m openonda source/solvers/fvm source/solvers/vpm source/coupler`:
  3,282 definitions, 612 without docstrings, and 81.4% documented. A separate
  AST inventory of public top-level functions/classes and public methods on
  public classes found 1,596 symbols, zero missing docstrings, and zero
  `docstring_parser` parse errors. The remaining interrogate misses are private
  helpers, generated Taichi kernels/accessors, and compatibility internals.
* `ruff check source/solvers/fvm source/solvers/vpm source/coupler source/_numba.py openonda tests`,
  `python -m compileall -q source openonda tutorials tests`, and `git diff --check`
  all passed.
* The full `pytest -q` run passed all 668 collected tests. This includes solver
  metadata creation/lifecycle coverage, tutorial schema/style checks, copied
  installed-tutorial execution, FVM/VPM/coupler numerical tests, and the
  read-only Numba cache fallback regression.
* `python install.py` rebuilt the wheel, installed it into the configured
  environment, passed `pip check`, and completed the installed-package verifier,
  including native mesher, FVM, plotting, direct tutorial-script, and Taichi
  smoke tests.
* Mypy and Pyrefly remain non-zero because the repository's existing dynamic
  third-party/Taichi interfaces and legacy typing debt are broader than this
  documentation pass. This is recorded rather than hidden. The final
  measurements are 599 Mypy errors across 51 of 275 checked source files and
  1,468 Pyrefly errors (91 suppressed); Ruff, syntax/import validation, and
  runtime tests are clean.

No numerical algorithm or solver equation was changed by the documentation
overhaul. Runtime changes are limited to solver-owned universal FVM/VPM metadata,
the documented tutorial/output cleanup, and a Numba import fallback that disables
the on-disk compilation cache only when Numba reports that no writable cache
locator exists.
