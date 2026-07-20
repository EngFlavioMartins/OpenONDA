# FVM performance and maintainability implementation plan

**Date:** 2026-07-20  
**Audience:** AI coding agent implementing the audited FVM improvements  
**Scope:** `source/solvers/FVM`, its tests, and FVM benchmark tooling  
**Implementation policy:** complete the low- and medium-risk work below; do not
perform the explicitly excluded high-risk numerical or architectural changes.

## 1. Objective

Improve the native FVM solver's initialization time, steady-state step time,
peak memory, and partitioned MPI scaling without changing its discretization,
boundary-condition meaning, convergence criteria, public scientific
configuration, or output fields.

The implementation is complete when:

1. The safe serial hot-path changes are implemented and measured.
2. Duplicate initialization work and avoidable Python-object construction are
   removed.
3. Momentum and pressure assembly retain fewer full-domain arrays and reuse
   static work.
4. Sparse-pattern and preconditioner caches have explicit ownership and
   cleanup.
5. Partitioned halo exchange uses precomputed numeric layouts.
6. Partitioned PETSc matrices, vectors, and KSP objects persist across solves.
7. Runtime/MPI choices remain behind `FVMSetup.cores`; tutorial and user
   `allrun.sh` files do not acquire MPI/backend configuration.
8. Existing numerical verification remains unchanged except for new
   characterization and performance tests.

This is an implementation plan, not permission to redesign the numerical
method. Every phase has a narrow behavioral contract and must be reviewable on
its own.

## 2. Starting evidence and performance baseline

The audit measured the current working tree on an Apple Silicon Mac with the
canonical OpenONDA Python 3.11 environment:

| Case | Initialization | One step | Linear work | Other work | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 10,000 cells | 1.13 s | 0.447 s | 0.096 s | 0.351 s | 216 MB |
| 99,856 cells | 11.19 s | 5.30 s | 3.36 s | 1.93 s | 1.00 GB |
| Existing 1M record | 242.7 s | 65.3 s | 19.7 s | 45.7 s | 3.62 GB |

The warmed 100k profile identified:

- approximately 54% of profiled step time in sparse solves;
- approximately 28% in Python slip/empty boundary projection;
- 599,136 `_remove_normal_component` calls plus 199,712 predicted-velocity
  empty-boundary iterations in one step;
- four pressure-face-conductance calculations where two are sufficient;
- pressure boundary descriptors rebuilt for each pressure corrector.

A focused 300,000-face projection benchmark measured 0.532 s for the Python
loop and 0.00774 s for vectorized NumPy, about a 69x kernel speedup.

The current weak-scaling benchmark at 10,240 cells/rank measured:

| Ranks | Initialization | Step | Peak rank RSS |
|---:|---:|---:|---:|
| 1 | 0.67 s | 0.221 s | 310 MB |
| 2 | 1.35 s | 0.245 s | 423 MB |
| 4 | 3.07 s | 0.306 s | 593 MB |

That benchmark starts from the exact uniform channel solution and reports zero
continuity error. It is useful for setup/communication overhead but is not a
valid Krylov-scaling qualification.

Validation state before implementation:

- 274 fast unit/verification tests pass.
- All 11 collective MPI/PETSc tests pass with two ranks.
- 14 of 15 slow non-OpenFOAM physics tests pass.
- The known failure is
  `test_ibm_square_force_and_wake_match_body_fitted_reference[0.125]`:
  11.4959% drag disagreement against a 3% limit. It reproduced exactly twice.
- `compileall` and Ruff `F` checks pass.

Do not hide, relax, mark `xfail`, or otherwise alter the known IBM failure as
part of this plan.

## 3. Mandatory agent rules

### 3.1 Repository safety

Before each phase:

1. Read the repository `AGENTS.md`.
2. Run `git status --short`.
3. Treat every pre-existing modification or untracked file as user-owned.
4. Do not reset, revert, delete, or reformat unrelated changes.
5. Do not edit tutorial `allrun.sh`/`allclean.sh` files for backend, PMIx,
   rank-count, or thread configuration.
6. Keep `FVMSetup.cores` as the user-facing parallel resource setting.

The worktree was changing during the audit. Never assume a file is unchanged
since this plan was written; inspect the live source before applying a task.

### 3.2 Change isolation

Use one commit or review unit per numbered phase. Do not combine:

- numerical behavior changes with performance refactors;
- MPI changes with serial assembly changes;
- dead-code cleanup with a numerically sensitive refactor;
- official baseline updates with solver changes.

If a phase requires changing a discretized equation, coefficient sign,
under-relaxation formula, pressure reference policy, convergence tolerance, or
boundary value, stop and report that the task crossed the high-risk boundary.

### 3.3 Required checks for every Python phase

Run, at minimum:

```bash
pyrefly check <changed Python files>
ruff check <changed Python files>
ruff format --check <changed Python files>
python -m compileall -q source/solvers/FVM
```

Pyrefly has a pre-existing repository baseline. Fix every new error introduced
by the phase; do not attempt a broad unrelated type cleanup.

Run targeted tests first, then the complete gates listed in Section 14.

### 3.4 Performance measurement rules

- Benchmark the exact same revision state, interpreter, environment variables,
  mesh, solver settings, and core count before and after a phase.
- Report initialization and warmed-step results separately.
- Use at least one warmup step and five measured steps for serial micro/step
  comparisons.
- Report median and minimum, not only a single observation.
- Record peak RSS and linear iteration/setup telemetry.
- A change must not be called faster based solely on cProfile time.
- Never commit a new official baseline from a dirty tree.
- Treat a regression greater than 5% in median runtime or peak RSS as a
  blocker unless the phase explicitly trades one for the other and documents
  the measured benefit.

## 4. Risk boundary

### Included

The following are low or medium risk and belong in this plan:

- benchmark correctness and warm/cold separation;
- vectorizing algebraically identical boundary operations;
- caching topology and boundary metadata with explicit invalidation;
- sharing already-computed Rhie--Chow intermediates;
- eliminating duplicate topology validation and duplicate LSQ construction;
- vectorizing geometry/validation with the same formulas;
- batching the existing LSQ matrix operations without changing the method;
- sharing the identical momentum matrix and preconditioner;
- reducing temporary arrays and adding solver-owned workspaces;
- bounded/solver-owned sparse caches;
- diagnostic cadence changes that retain mandatory health checks;
- precomputed numeric MPI halo schedules;
- persistent PETSc objects with unchanged solver methods/tolerances;
- sequential/streamed root partition payload construction;
- fail-fast rejection of unqualified partitioned dynamic Smagorinsky;
- dead-code and stale-comment cleanup after behavior is stable.

### Explicitly excluded high-risk work

Do not implement any of the following in this plan:

1. Changing the Rhie--Chow formula, pressure conductance, pressure constraint,
   or pressure/velocity correction signs.
2. Changing when non-orthogonal sweeps update pressure, velocity, or flux.
3. Fixing or tuning the IBM force algorithm, marker kernel, forcing loop,
   pressure coupling, force integration, or validation tolerance.
4. Replacing LSQ normal equations with a weighted-design QR/SVD method, using
   active-dimensional 2-D LSQ, or otherwise changing reconstructed gradients.
5. Adding GPU/CUDA/Metal/Vulkan execution.
6. Adding graph partitioning, changing cell ownership, or implementing a
   distributed mesh reader.
7. Changing default scientific schemes, tolerances, number of correctors,
   under-relaxation factors, or direct/iterative solver semantics.
8. Removing public `ExecutionConfig` symbols or making other breaking API
   changes.
9. Enabling asynchronous MPI output.
10. Changing dynamic-mesh/ALE behavior.

For non-orthogonal correction and IBM, this plan permits tests and
characterization only. Any discovered numerical failure becomes a separate
human-reviewed task.

### Phase dependency map

Implement the phases in the following order unless a dependency is explicitly
satisfied by the live tree:

| Phase | Prerequisite | May overlap in review with |
|---|---|---|
| 0 — instrumentation | none | documentation-only work |
| 1 — boundary projection | Phase 0 baseline | Phase 2 or 3 if files do not overlap |
| 2 — pressure workspace | Phase 0 baseline | Phase 1 or 3 if files do not overlap |
| 3 — initialization/LSQ | Phase 0 baseline | Phase 1 or 2 if files do not overlap |
| 4 — momentum workspaces | Phases 1–3 merged and rebaselined | none |
| 5 — cache/diagnostic lifecycle | Phases 2–4 | independent output-only commits |
| 6 — numeric MPI data path | Phase 0 and relevant Phase 3 metadata | none |
| 7 — persistent PETSc path | Phases 4 and 6 | none |
| 8 — qualification/cleanup | all applicable prior phases | documentation-only work |

Do not parallel-edit the same solver modules. After any concurrent,
non-overlapping review units are merged, rerun the Phase 0 baseline before
starting the next dependent phase. Serial work should be qualified before MPI
work so that MPI measurements do not conceal local regressions.

## 5. Phase 0 — repair and extend performance instrumentation

**Risk:** low  
**Purpose:** create trustworthy measurements before changing hot paths.

### Files

- `scripts/benchmarks/benchmark_fvm.py`
- `scripts/benchmarks/benchmark_fvm_mpi.py`
- `docs/benchmarks/`
- new benchmark tests under `tests/fvm/` if needed

### Tasks

1. Extend the serial benchmark CLI with:

   - `--warmup-steps`, default `1`;
   - `--measured-steps`, default `5`;
   - an explicit cold one-step mode for compatibility;
   - separate initialization, first-step, warmed-step, linear-setup,
     linear-solve, operator/diagnostic, and peak-RSS fields;
   - median, minimum, maximum, and per-step samples;
   - the number of linear iterations and preconditioner rebuilds.

2. Preserve the existing schema reader or increment the schema version and add
   an explicit compatibility parser. Do not silently reinterpret old JSON.

3. Make baseline comparison reject a dirty source tree by default. Add an
   explicit development-only override such as `--allow-dirty`; include
   `"official": false` in such output.

4. Split backend qualification into:

   - cold runtime/startup cost;
   - warmed steady-state step cost;
   - numerical parity;
   - peak RSS.

   This prevents Numba or Taichi initialization from being mixed with ordinary
   step throughput.

5. Replace the MPI benchmark's exact steady initial state. The simplest
   acceptable first version is zero initial velocity with a nonzero inlet,
   followed by enough warmup steps to produce:

   - nonzero pressure RHS;
   - at least one pressure iteration;
   - at least one momentum iteration or a documented converged initial guess;
   - nonzero but finite continuity telemetry before correction.

   If this does not exercise every rank meaningfully, add a deterministic
   distributed initial-field helper in benchmark code only. Do not add a
   public solver API solely for benchmarking.

6. Record MPI timing components separately:

   - root mesh/geometry;
   - partition construction;
   - payload distribution;
   - halo exchange;
   - PETSc matrix/vector/KSP setup;
   - PETSc solve;
   - diagnostics.

7. Add a repeated-solver memory test that creates and closes several distinct
   meshes in one process and checks that retained cache memory does not grow
   without bound. Initially mark it as a characterization test if the current
   global caches fail.

8. Capture baseline artifacts in `/tmp` or another untracked location for:

   - 10k and 100k serial;
   - 10,240 cells/rank at 1, 2, and 4 ranks;
   - boundary projection microbenchmark;
   - repeated solver construction.

### Acceptance

- The benchmark produces deterministic schema-valid JSON.
- A deliberately exact/trivial MPI case is detected or clearly labeled.
- Warmed measurements exclude initialization and first-use compilation.
- No official JSON under `docs/benchmarks` is changed yet.
- Benchmark unit tests pass without requiring timing thresholds on shared CI.

### Commit boundary

Commit benchmark/test tooling alone. Do not include solver changes.

## 6. Phase 1 — vectorize velocity boundary projection

**Risk:** low  
**Expected benefit:** largest simple serial hot-path reduction, especially for
pseudo-2D meshes.

### Files

- `source/solvers/FVM/assemble/momentum.py`
- `source/solvers/FVM/solve/simple_solver.py`
- boundary-specific tests under `tests/fvm/`

### Tasks

1. Replace the Python loop in `_apply_empty_bc_ustar` with patch-array
   operations:

   - gather owner velocity once;
   - compute `|Sf|` along axis 1;
   - divide safely to form patch normals;
   - compute normal velocity with `einsum` or `sum`;
   - assign all boundary ghosts at once;
   - preserve the current degenerate-face fallback by copying owner values
     wherever `|Sf| <= 1e-10`.

2. Replace the loop in `_apply_slip_bc` with the same vector expression.
   Preserve exact ghost indices and owner ordering.

3. Keep `_remove_normal_component` temporarily if tests or external internal
   imports use it. After an `rg` reference check, delete it only in the cleanup
   phase.

4. Do not cache a full `n_faces x 3` normal array globally. That would add
   approximately 96 MB at four million faces. If caching is useful, cache
   boundary-patch normals only in the later boundary-layout workspace.

5. Add focused tests for:

   - empty, slip, and symmetry patches;
   - arbitrary non-axis-aligned normals;
   - multiple faces with noncontiguous owners;
   - degenerate-vector fallback at the private helper level;
   - equivalence with the old scalar formula to `rtol=0`, `atol` appropriate
     for re-ordered floating operations;
   - no normal component after projection;
   - tangential component preservation.

6. Add or retain an assertion that geometry validation rejects actual
   zero-area production faces; helper fallback is defensive only.

### Acceptance

- No per-face Python loop remains in either projection path.
- cProfile reports zero `_remove_normal_component` calls during the 100k step.
- The projection microbenchmark improves by at least 20x.
- The warmed 100k pseudo-2D step target is at least 15% faster.
- Peak RSS does not increase by more than 2%.
- All fast FVM and boundary tests pass.

### Commit boundary

Commit only vectorization and its tests.

## 7. Phase 2 — pressure-correction workspace and boundary layout

**Risk:** medium  
**Purpose:** stop recomputing identical dynamic and static pressure data.

### Files

- `source/solvers/FVM/solve/simple_solver.py`
- `source/solvers/FVM/solve/pimple_solver.py`
- optionally a new internal module such as
  `source/solvers/FVM/solve/boundary_runtime.py`
- `tests/fvm/test_rhie_chow_consistency.py`
- pressure/boundary/PIMPLE tests

### Design

Introduce internal data containers; suggested names:

```python
@dataclass(frozen=True)
class PressureBoundaryLayout:
    signature: tuple
    face_indices: np.ndarray
    type_codes: np.ndarray
    patch_slices: tuple[slice, ...]


@dataclass
class PressureCorrectionWorkspace:
    DU: np.ndarray
    face_conductance: np.ndarray
```

The exact names are flexible. The important properties are:

- topology/type arrays are immutable and cached;
- boundary values remain dynamic;
- `DU` and conductance are produced once per pressure assembly and passed to
  the corresponding correction;
- standalone legacy callers continue to work.

### Tasks

1. Split `_build_boundary_face_arrays` into:

   - a structural layout builder that fills face indices and type codes by
     patch-level vector assignment;
   - a dynamic value updater that reads current `value_p`,
     `value_p_field`, freestream state, and replay/coupling overrides.

2. Define a boundary signature containing at least:

   - patch start/count;
   - resolved pressure strategy;
   - any structural external-override mode.

   Rebuild the layout if this signature changes. Do not assume coupling code
   never changes a boundary type at runtime.

3. Build the layout once in `SIMPLESolver.__init__` or lazily on first use.
   Store it on the algorithm instance, not in a module-global cache.

4. Extend pressure assembly with an internal opt-in return for correction
   data. Preserve the existing three-value return for standalone callers and
   tests. Acceptable patterns:

   - `return_workspace=False`;
   - a new internal `assemble_pressure_correction_system` used by PIMPLE, with
     the old function as a compatibility wrapper.

5. Pass the assembly's exact `DU` and `face_conductance` to
   `correct_velocity_and_flux`. If no workspace is supplied, retain current
   recomputation for compatibility.

6. Remove `phi_star.copy()` in PIMPLE only after proving that pressure assembly
   returns a newly owned flux array and no later consumer needs the
   uncorrected value.

7. Replace any boundary-face Python copy-back loop with vectorized indexed
   assignment.

8. Precompute geometry-only conductance terms once per static mesh if they can
   be represented without a large redundant vector field. Suitable cached
   scalars include:

   - `|Sf|`;
   - `|CF|`;
   - `(Sf · e) / |CF|`;
   - squared normal components if memory measurement justifies them.

   Prefer recomputing a cheap vector expression over adding hundreds of
   megabytes of permanent state. Measure the tradeoff at 1M-equivalent sizes.

9. Add identity tests that monkeypatch or count
   `_compute_pressure_face_conductance` and confirm exactly one call per
   assembly/correction pair.

10. Retain the existing skewed-mesh assembly/correction conductance test. Add
    an assertion that the shared workspace and compatibility path yield the
    same corrected `U` and `phi`.

### Acceptance

- Standard PIMPLE with two pressure correctors calls the conductance builder
  twice, not four times.
- Boundary topology arrays are built once unless their signature changes.
- Dynamic boundary values remain visible immediately.
- Corrected velocity, pressure, flux, continuity, and residual telemetry match
  the compatibility path within existing tolerances.
- Warmed 100k step shows no regression; target improvement is 2–5%.
- Peak RSS does not increase by more than 2%.

### Commit boundary

Commit pressure workspace/layout and focused tests only.

## 8. Phase 3 — remove duplicate initialization and vectorize validation

**Risk:** medium  
**Expected benefit:** substantial initialization reduction and lower peak
Python-object memory.

### Files

- `source/solvers/FVM/core/solver.py`
- `source/solvers/FVM/mesh/geometry.py`
- `source/solvers/FVM/mesh/validation.py`
- `source/solvers/FVM/fields/gradients.py`
- `source/solvers/FVM/mesh/partition.py`
- initialization, cyclic, LSQ, topology, and mesh-quality tests

### 8.1 Initialization ordering

Refactor serial initialization to:

```text
load/set mesh
→ validate topology once
→ compute base geometry without LSQ
→ install configured boundary conditions
→ configure cyclic topology/geometry
→ compute LSQ once if requested
→ validate geometry and quality
→ prepare assembly workspaces
```

Tasks:

1. Separate `validate_topology` and `validate_geometry` calls in `Solver`.
   Do not call the combined `validate_mesh` twice.

2. Add an explicit way for `compute_mesh_geometry` to compute base geometry
   without automatically constructing LSQ. Do not overload `"gauss"` in a way
   that incorrectly records the requested scheme.

3. Configure cyclic boundaries before LSQ so periodic neighbor entries are
   present in the only LSQ build.

4. Preserve mesh-quality reporting, including LSQ condition/rank statistics,
   after LSQ has been attached.

5. Add a test that counts `compute_lsq_geometry` calls during serial cyclic
   solver construction and asserts exactly one.

### 8.2 Partitioned ordering

The partitioned route currently computes global LSQ and then recomputes local
LSQ for every payload. Change it to:

```text
rank 0 topology validation
→ rank 0 base geometry only
→ base geometry validation
→ localize topology/base geometry
→ scatter
→ install local configured boundaries
→ local LSQ once per rank
→ reduce LSQ quality metrics
```

Partitioned cyclic patches remain unsupported; do not add them.

Tasks:

1. Ensure `localize_mesh_and_geometry` skips global `lsq_*` arrays and receives
   a correct requested-gradient flag.

2. Compute local LSQ only after local boundary dictionaries have their
   configured solver types.

3. Report global LSQ quality with reductions:

   - maximum finite condition;
   - sum of rank-deficient owned cells;
   - sum of SVD-owned cells.

   Count owned cells only, never halos.

4. Test that partitioned LSQ still matches the replicated reference and that
   each rank constructs LSQ once.

### 8.3 Topology and geometry validation

1. For fixed-width ndarray faces:

   - validate `ndim`, minimum width, index bounds, and repeated vertices in
     batches;
   - use sorted rows or pairwise comparisons without calling `np.unique` once
     per face.

2. Keep the generic list-of-arrays fallback for polyhedra. Do not remove mesh
   format support.

3. Replace `cell_face_distances = [[] for ...]` with numeric reductions:

   - initialize cell minima to `+inf`, maxima to zero;
   - compute owner-face distances vectorized;
   - include neighbor-face distances for interior faces;
   - accumulate with `np.minimum.at` and `np.maximum.at`;
   - verify every cell received at least one value;
   - compute aspect ratio from the reduced arrays.

4. Vectorize secondary interior and boundary face geometry in
   `compute_geometry`:

   - bulk owner/neighbour centroid gathers;
   - bulk `CF`, owner-to-face and neighbor-to-face vectors;
   - bulk weights with guarded division;
   - bulk wall distance and limited distance.

5. Add chunking for the largest padded face/cell tensors if peak RSS remains
   high. Chunking must preserve the exact formulas and face/cell ordering. Use
   a private constant or internal heuristic, not a user-facing tuning option.

6. Add parity tests comparing vectorized/chunked output to the existing
   formulas on:

   - orthogonal hexahedra;
   - sheared hexahedra;
   - tetrahedra;
   - prisms;
   - mixed/polyhedral faces;
   - periodic meshes.

### 8.4 Batch the existing LSQ calculations

This subtask may optimize the implementation but must not change the method.

1. Replace per-cell Python stencil lists with flat arrays built from:

   - owner→neighbor interior contributions;
   - neighbor→owner interior contributions;
   - non-empty boundary/paired contributions.

2. Produce the same `lsq_nei_phi_idx`, `lsq_owner_cell`,
   `lsq_nei_w2_dr`, and `lsq_sum_w2dr` contract.

3. Assemble each cell's existing 3x3 normal matrix by numeric reductions.

4. Use batched `np.linalg.cond`, `matrix_rank`, QR/solve, and `pinv` where the
   installed NumPy supports the same behavior. Split full-rank and fallback
   cell indices so each batch uses the same current decision threshold
   (`1e8`).

5. Do not change:

   - inverse-distance weights;
   - rank threshold;
   - condition threshold;
   - boundary inclusion;
   - 2-D pseudoinverse behavior;
   - returned gradient convention.

6. Compare old and new LSQ arrays and reconstructed scalar/vector gradients on
   deterministic meshes before deleting the old path. A temporary
   test-only/reference implementation is acceptable.

### Acceptance

- Serial cyclic initialization calls topology validation once and LSQ once.
- Partitioned initialization never computes global LSQ.
- No Python list per cell is created during aspect-ratio validation.
- No per-face Python loop remains in secondary face geometry for ndarray
  connectivity.
- Initialization target: at least 25% faster at 100k LSQ cyclic cells.
- Peak initialization RSS target: at least 10% lower at 100k, with no increase
  at 1M-equivalent geometry.
- Geometry, quality metrics, LSQ gradients, cyclic parity, and physics
  verification pass.

### Commit boundaries

Prefer separate commits for:

1. initialization ordering;
2. validation vectorization;
3. secondary geometry vectorization/chunking;
4. LSQ flat/batched construction.

Do not proceed to the next subtask if the earlier one changes numerical
results outside existing tolerances.

## 9. Phase 4 — momentum matrix and temporary-memory reuse

**Risk:** medium  
**Purpose:** represent the mathematical three-component system as one matrix
with three RHS vectors.

### Files

- `source/solvers/FVM/assemble/momentum.py`
- `source/solvers/FVM/assemble/diffusion.py`
- `source/solvers/FVM/assemble/convection.py`
- `source/solvers/FVM/assemble/matrix_assembly.py`
- `source/solvers/FVM/solve/pimple_solver.py`
- momentum, MMS, transient, source, and solver telemetry tests

### Design

Add a new internal result such as:

```python
@dataclass
class MomentumSystem:
    matrix: csr_matrix
    rhs: np.ndarray              # (n_owned_or_cells, 3)
    relaxed_diagonal: np.ndarray # (n_cells,)
```

Keep a compatibility wrapper for tests or consumers that still need the old
component dictionary. The production PIMPLE path must use the compact form.

### Tasks

1. Prove with assertions/tests that implicit convection, diffusion, transient,
   source-implicit, and relaxation coefficients are component-independent.
   If a component-dependent implicit source exists, stop this phase and report
   the assumption failure.

2. Construct the common implicit face coefficients and CSR matrix once.

3. Compute the three explicit RHS columns without retaining three full sets of
   diffusion, convection, and combined flux dictionaries.

4. Compute invariant diffusion geometry once per momentum assembly and reuse it
   for all components. Keep it ephemeral unless profiling proves a
   solver-persistent cache has a worthwhile time/memory tradeoff.

5. Create reusable face buffers in a solver-owned `MomentumAssemblyWorkspace`.
   Buffers may be overwritten after their contribution has been reduced into
   the matrix/RHS.

6. Add transient and implicit-source diagonal terms once.

7. Apply under-relaxation once to the shared matrix diagonal. Build the three
   relaxed RHS columns by broadcasting the shared diagonal.

8. Direct SciPy path:

   - retain one factorization/three-column RHS behavior;
   - preserve per-component residual telemetry;
   - charge setup/solve time to the first component only, as today.

9. Iterative SciPy path:

   - solve the three RHS columns sequentially against one matrix;
   - reuse one ILU/preconditioner where configured;
   - use current `U[:n, component]` as the initial guess for outer correctors;
   - retain `U_old` only as the transient history, not as a forced Krylov
     initial guess.

10. PETSc path:

    - reuse the same matrix workspace for all momentum components;
    - persistent KSP work is implemented in Phase 7, but the compact momentum
      interface must make it possible.

11. Keep `A_U` externally compatible as `(n_cells, 3)` for pressure correction.
    Populate it from the shared diagonal. Do not change the Rhie--Chow
    coefficient formula.

12. Remove stored `H` and `grad_p_comp` only after `rg` and tests prove there
    are no consumers.

13. Avoid expanding scalar `rho` and laminar scalar `nu` into cell arrays:

    - add scalar-aware validation;
    - compute scalar face values without `np.full`;
    - retain the array path for turbulent/variable viscosity;
    - retain variable-density behavior even if the current config normally
      supplies a scalar.

14. Add tests comparing compact and compatibility assemblies:

    - matrix structure and values;
    - all three RHS columns;
    - relaxed diagonal;
    - direct and iterative solutions;
    - source-explicit/source-implicit behavior;
    - BDF1 and BDF2;
    - fixed, freestream, cyclic, empty, slip, and wall boundaries;
    - partitioned owned solutions.

### Acceptance

- Production PIMPLE retains one momentum CSR matrix, not three.
- Direct solve still performs one factorization for three RHS columns.
- Iterative solves reuse one eligible preconditioner.
- The number of full-face coefficient buffers is materially reduced and
  documented.
- 100k peak RSS target: at least 10% lower than the Phase 3 baseline.
- Warmed step does not regress; target improvement is 5–10% outside linear
  solves.
- MMS, temporal-order, source, cyclic, and partitioned parity tests pass.

### Commit boundary

Commit compact assembly/workspace and tests without cache-lifecycle or PETSc
persistence changes.

## 10. Phase 5 — cache ownership, diagnostics, and output overhead

**Risk:** low to medium

### 10.1 Sparse-pattern cache

Files:

- `source/solvers/FVM/assemble/matrix_assembly.py`
- `source/solvers/FVM/core/solver.py`

Tasks:

1. Remove the unused `_ILU_CACHE` in `matrix_assembly.py`.

2. Replace module-global `_SPATIAL_CACHE` ownership with one of:

   - mesh-owned cached patterns keyed by `include_boundaries`; or
   - solver-owned matrix assembly workspaces.

3. Do not retain topology `.tobytes()` in a process-global dictionary.
   A mesh-local topology version/identity is enough after validation because
   topology mutation is unsupported during a static solve.

4. Define explicit invalidation if cyclic topology is configured after base
   mesh creation. Pattern construction must occur after cyclic setup.

5. Ensure closing and deleting a solver releases patterns when no mesh/solver
   references remain.

### 10.2 ILU/AMG cache

Files:

- `source/solvers/FVM/solve/linear_interface.py`
- algorithm/solver initialization and close paths

Tasks:

1. Add a solver-owned `LinearSolverCache` or equivalent.

2. Store ILU and AMG entries under equation-family keys plus matrix shape and
   topology version. Avoid hashing/copying full CSR pattern bytes every solve.

3. Preserve diagonal-change rebuild policies and failure semantics.

4. Clear/destroy the cache in `Solver.close`.

5. Keep a compatibility transient cache path only if standalone
   `solve_linear_system` tests require it. Bound that cache and provide an
   explicit clear function.

6. Extend repeated-solver tests to verify that distinct million-cell-equivalent
   topology entries are not retained after close/GC.

### 10.3 Diagnostics

Files:

- `source/solvers/FVM/core/solver.py`
- `source/solvers/FVM/fields/diagnostics.py`
- output/diagnostic config and tests

Tasks:

1. Split mandatory health diagnostics from extended diagnostics.

   Mandatory every solve:

   - nonfinite checks;
   - linear convergence/failure;
   - continuity;
   - acceptance-policy metrics required for abort decisions;
   - CFL when adaptive stepping or a CFL acceptance limit requires it.

   Extended at output/diagnostic cadence:

   - vorticity;
   - enstrophy;
   - full extrema not required by acceptance;
   - y+;
   - force gradients when forces are not scheduled.

2. Preserve `last_diagnostics` schema. For cadence-skipped extended values,
   either carry the most recently computed value with an explicit sample step,
   or keep current computation if changing schema would be breaking. Do not
   silently present stale values as current.

3. In coupled Picard calls, avoid recomputing extended physical-step
   diagnostics on every inner `solve_pimple` unless acceptance requires them.
   Add an internal `diagnostic_level` selected by the coupler/solver, not a
   tutorial shell option.

4. Reuse `_courant_field` in adaptive timestep handling rather than computing
   Courant twice for the same solved state.

5. Compute y+ only when:

   - y+ patches exist; and
   - its logging/output cadence is due.

6. Preserve derived-field caching across force and VTK output for the same
   solved state.

### 10.4 Output

1. Retain bounded serial asynchronous snapshots.
2. Keep the required field copies that make background snapshots immutable.
3. Avoid computing vorticity/Courant twice for diagnostics plus output.
4. Do not enable threaded partitioned output.
5. Avoid a duplicate partition visualization mesh where this is safe:

   - for fixed-width structured hexahedra, reuse the compact local
     `cell_vertices`/point topology, which already describes complete
     owned-plus-halo cells;
   - for generic polyhedra, retain the current visualization topology unless
     the local payload contains enough complete cell topology to reconstruct it
     later without access to the global mesh;
   - do not make manual output fail merely because `auto_write` was disabled
     after solver construction.

   A truly lazy generic-polyhedral visualization mesh is permitted only if its
   source topology has explicit ownership and measured memory is lower than
   the eager representation.

### Acceptance

- Repeated solver construction has bounded retained memory.
- Cache entries are released on close.
- No full-pattern bytes are generated on each AMG lookup.
- Mandatory acceptance behavior is unchanged.
- Coupled inner solves and no-output runs avoid extended diagnostics.
- Serial/partitioned output and restart tests pass.
- Warmed operator/diagnostic time is lower with no physics-field change.

## 11. Phase 6 — numeric MPI halo schedule and streamed partition payloads

**Risk:** medium

### Files

- `source/solvers/FVM/mesh/partition.py`
- `source/solvers/FVM/core/parallel.py`
- `source/solvers/FVM/core/solver.py`
- partition, coupling, output, and MPI tests

### 11.1 Precomputed halo indices

Extend `HaloSchedule` to own:

- neighbor rank list only, not `size` empty entries;
- send local indices;
- receive local indices;
- send/receive counts and displacements;
- reusable contiguous send/receive buffers per dtype/trailing shape, or a
  bounded buffer cache.

Tasks:

1. Compute global→local mappings once during `CellPartition.__post_init__`.
2. Convert `send_global_ids` and `receive_global_ids` to local index arrays
   once.
3. Remove dictionary/list-comprehension mapping from every exchange.
4. Replace Python-object `comm.alltoall` with numeric communication:

   - preferred: distributed-graph communicator plus `Neighbor_alltoallv`;
   - acceptable: deterministic nonblocking `Irecv`/`Isend` for neighbor ranks
     followed by `Waitall`.

5. Support scalar, vector, and tensor trailing shapes used by:

   - pressure;
   - velocity;
   - `A_U`;
   - gradients;
   - turbulence viscosity.

6. Use explicit MPI datatypes derived from NumPy dtype. Reject unsupported
   object/noncontiguous arrays rather than pickling them.

7. Preserve an optional object-exchange reference implementation in tests
   until parity is established, then remove it from production.

8. Add MPI tests for:

   - uneven ownership;
   - ranks with no neighbor on one side;
   - scalar/vector/tensor values;
   - two and four ranks;
   - repeated exchanges without allocation growth;
   - exact global-cell-field reconstruction.

### 11.2 Stream root payloads

The global mesh may remain on rank zero in this plan, but rank zero must not
hold all localized payloads simultaneously.

Tasks:

1. Replace `payloads = [localize(...) for rank in ...]` plus object scatter
   with a deterministic protocol:

   - root builds its own payload;
   - root builds one remote payload at a time and sends it;
   - remote ranks receive exactly one payload;
   - errors are broadcast before ranks enter incompatible collectives.

2. Release each remote payload before constructing the next.

3. Separate partition metadata/topology arrays from large geometry arrays if
   chunked numeric sends are straightforward. Object send is acceptable for
   one-at-a-time payloads in this phase; do not invent a new mesh file format.

4. Apply the safe visualization policy from Phase 5:

   - do not create a second visualization mesh for structured partitions when
     local `cell_vertices` are already complete;
   - keep generic-polyhedral visualization data in the one-at-a-time payload
     unless a tested compact/lazy representation is available;
   - include visualization bytes in payload and root-memory telemetry.

5. Add initialization telemetry for:

   - global geometry;
   - per-rank localization;
   - distribution;
   - visualization topology construction or reuse.

6. Test root and non-root exceptions so a failed localization cannot leave
   other ranks hung.

### Acceptance

- Steady halo exchange creates no rank-count-length Python payload list.
- No pickling occurs for numeric halo fields.
- Repeated exchange allocations plateau.
- Rank-zero localization retains no more than one remote payload at once.
- Weak-scaling root RSS grows more slowly than the Phase 0 baseline.
- All collective MPI, coupling, checkpoint, and partitioned output tests pass.

### Commit boundaries

Use separate commits for numeric halo exchange and streamed partition/
visualization construction.

## 12. Phase 7 — persistent partitioned PETSc workspaces

**Risk:** medium-high but included because it preserves the linear systems and
is central to partitioned performance.

### Files

- `source/solvers/FVM/solve/petsc_partitioned.py`
- `source/solvers/FVM/solve/linear_interface.py`
- `source/solvers/FVM/assemble/matrix_assembly.py`
- `source/solvers/FVM/assemble/momentum.py`
- solver/algorithm lifecycle
- PETSc MPI tests and benchmark

### Design

Add a solver-owned workspace per equation family:

```text
momentum workspace:
    PETSc Mat
    RHS Vec
    solution Vec
    residual Vec only when telemetry requires it
    KSP + PC
    topology/ownership signature

pressure workspace:
    same, plus optional constant nullspace
```

### Tasks

1. Introduce `PartitionedLinearWorkspace` with explicit `close`/`destroy`.

2. Build the PETSc matrix once per topology/ownership layout:

   - preallocate diagonal/off-diagonal nonzeros from owned CSR rows;
   - preserve global row/column numbering;
   - verify the PETSc ownership range;
   - install nullspace once where applicable.

3. Stop assembling equations for halo rows:

   - add an owned-row assembly pattern for partitioned execution with shape
     `(n_owned, n_local)` before global-column mapping;
   - retain contributions from every local face that touches an owned row;
   - omit contributions whose destination row is a halo cell;
   - assemble only the first `n_owned` RHS entries;
   - compare the owned-row matrix and RHS exactly against
     `full_local_matrix[:n_owned]` and `full_local_rhs[:n_owned]` before making
     it the production path;
   - retain full square local assembly for serial/replicated execution and as a
     temporary MPI reference in tests.

   Processor faces can have a ghost owner and owned neighbor in the localized
   orientation. The row filter must be based on destination row ownership, not
   on an assumption that the face owner is locally owned.

4. Update `OwnedRowsCSR.from_local` or add a new constructor that accepts the
   compact owned-row/local-column representation without first slicing and
   copying a square local CSR matrix. Map local columns to global IDs once per
   topology.

5. Replace the Python `for local_row ... setValues` loop with the fastest
   supported CSR insertion route in the installed petsc4py version. Validate
   whether `setValuesCSR` supports the local-owned/global-column layout. If it
   does not, use batched row blocks rather than per-row Python calls.

6. For each solve:

   - zero/update matrix numeric values;
   - update RHS values;
   - update initial guess values;
   - assemble;
   - reuse KSP/PC according to the configured policy;
   - solve;
   - copy owned solution once;
   - exchange halo through Phase 6.

7. Keep separate invalidation conditions:

   - topology/ownership/method/nullspace change: rebuild workspace;
   - coefficient change: update numeric matrix;
   - preconditioner rebuild: follow method-specific policy and record it.

8. Momentum:

   - update the matrix once;
   - solve three RHS components through the same workspace;
   - do not rebuild KSP/PC for y and z.

9. Pressure:

   - reuse Mat/KSP across correctors and timesteps;
   - update coefficients and RHS;
   - retain the constant nullspace object;
   - remove the RHS nullspace component each solve as today.

10. Correct telemetry:

   - compute actual initial residual from the supplied initial guess;
   - report setup time only when setup/rebuild occurs;
   - distinguish matrix-value update, PC rebuild, and KSP solve;
   - preserve final residual and convergence-reason checks.

11. Destroy PETSc objects in `Solver.close`, context-manager exit, and failed
   construction cleanup. Destruction must be collective where PETSc requires
   it.

12. Retain the current one-shot `solve_owned_rows` as a test/reference wrapper
    around a temporary workspace. Production PIMPLE must use persistent
    workspaces.

13. Add tests that count PETSc object creation:

    - one momentum Mat/KSP per solver topology;
    - one pressure Mat/KSP per solver topology;
    - no additional creation across ordinary correctors/timesteps;
    - expected rebuild after an explicit topology/signature change;
    - all objects destroyed on close.

14. Compare partitioned solutions against replicated SciPy/PETSc for:

    - pressure with fixed reference;
    - constant nullspace;
    - three momentum components;
    - LSQ and Gauss gradients;
    - multiple timesteps;
    - checkpoint/restart.

### Acceptance

- A standard two-corrector PIMPLE step no longer creates five PETSc
  Mat/KSP/PC stacks.
- Partitioned assembly produces owned rows directly and does not allocate or
  fill halo equation rows.
- Momentum uses one matrix/KSP setup for all components.
- Pressure reuses its matrix/KSP across correctors.
- Actual initial residual telemetry is correct.
- Partitioned-vs-replicated field tolerances remain unchanged.
- Nontrivial MPI benchmark target:

  - setup time reduced materially, preferably at least 50%;
  - warmed step improves at 2 and 4 ranks;
  - no increase in Krylov iterations caused by stale preconditioners;
  - memory plateaus across repeated steps.

### Stop condition

If persistent PC reuse changes convergence or field results and cannot be
resolved solely through an explicit coefficient-change rebuild policy, retain
persistent Mat/Vec/KSP allocation but rebuild the PC every solve. Do not loosen
tolerances or accept worse residuals to claim speedup.

## 13. Phase 8 — safety guards, artifact cleanup, and final qualification

**Risk:** low, after earlier phases are stable.

### 13.1 Partitioned dynamic Smagorinsky safety

The model currently calls its coefficient globally averaged but performs no
MPI global reduction and includes local halo cells.

Do not implement the distributed model in this plan. Instead:

1. Reject dynamic Smagorinsky when `parallel_mode="petsc_partitioned"` with a
   clear `NotImplementedError`.
2. Permit serial and replicated reference modes.
3. Add configuration tests proving the fail-fast behavior.
4. Do not reject WALE, sigma, or static Smagorinsky unless a separate parity
   test proves they are invalid.

### 13.2 Non-orthogonal characterization

Add test-only coverage without changing the algorithm:

1. Manufactured pressure/velocity case on a family of affine-sheared meshes.
2. Compare `n_orthogonal_correctors` 0, 1, and 2.
3. Record discretization error, continuity and pressure/velocity corrections,
   not only finite algebraic residuals.
4. If extra sweeps worsen or fail convergence, report a blocked high-risk
   algorithm task. Do not change update ordering here.

### 13.3 AI/refactor artifact cleanup

After `rg` reference checks and targeted tests:

1. Remove the meaningless `if True:` in `config/types.py`.
2. Remove no-op expressions in gradients and time integration.
3. Remove the unused `matrix_assembly._ILU_CACHE`.
4. Remove dead `_compute_geometric_diffusion` and
   `_process_interior_face_rhie_chow` only if no live/reference consumer
   remains.
5. Remove or explicitly adopt `mesh/cache.py::WeightCache`; do not retain an
   unused abstraction that defensively copies all geometry.
6. Make `core/operators.py` real or simplify it:

   - preferred: route matrix/RHS reductions through the `DiscreteOperators`
     instance;
   - alternative: replace it with an internal validated backend name;
   - preserve backend parity tests and public compatibility.

7. Update stale BDF2/Phase-2 comments in `test_temporal_order.py`.
8. Correct comments claiming partitioned assembly already creates only owned
   rows if implementation still assembles halo rows before slicing.
9. Keep replay/coupling overrides functional. Moving them into a dedicated
   override object is optional and must be a separate commit with replay
   parity tests.

### 13.4 Runtime configuration boundary

1. Keep normal applications on:

   ```python
   setup = FVMSetup(cores=N, ...)
   solver = setup_fvm_solver(setup, ...)
   ```

2. Ensure `_runtime_setup` continues to choose PETSc partitioned execution,
   synchronous MPI output, and internal thread caps.
3. Ensure the launcher uses the canonical absolute `sys.executable`, never
   bare `python`.
4. Do not add MPI flags or PMIx variables to tutorials.
5. Update user documentation to identify `ExecutionConfig` backend fields as
   expert/internal compatibility controls, while preserving the symbols.
6. Keep scientific schemes and tolerances user-visible because they affect
   the computed solution.

### 13.5 Final performance artifacts

After all accepted phases:

1. Run benchmarks from a clean tree at a named revision.
2. Generate new schema-versioned serial, backend and MPI reports.
3. Do not overwrite historical JSON; add dated/revisioned artifacts or update
   an index that identifies the qualified current baseline.
4. Include:

   - host and dependency identity;
   - warm/cold distinction;
   - source revision and clean status;
   - 10k, 100k, and available 1M result;
   - nontrivial 1/2/4-rank scaling;
   - initialization and peak memory;
   - linear iterations/rebuilds;
   - continuity and CFL parity.

5. Write a short completion report mapping each result to the phase that
   produced it and listing deferred high-risk findings.

## 14. Test matrix

Run targeted tests after each subtask. Before completing any phase that changes
solver Python, run the relevant rows below.

### Fast serial gate

```bash
conda run -n OpenONDA python -m pytest -q tests/fvm \
  -m "(unit or verification) and not slow and not mpi and not openfoam"
```

Expected starting result: 274 passed.

### Coupler gate

Required for changes to boundary layouts, repeated `solve_pimple`, partition
fields, or replay overrides:

```bash
conda run -n OpenONDA python -m pytest -q tests/coupler \
  -m "not mpi and not openfoam"
```

### Collective MPI/PETSc gate

Use the canonical environment launcher and Python. The exact path should come
from the active environment, not be hardcoded in product scripts.

```bash
conda run --no-capture-output -n OpenONDA bash -lc \
  'PMIX_MCA_pif_base_retain_loopback=1 \
   "$CONDA_PREFIX/bin/mpiexec" -n 2 \
   "$CONDA_PREFIX/bin/python" -m pytest -q tests/fvm/test_petsc_parallel.py'
```

Expected starting result: all 11 collective tests pass on both ranks.

Run four-rank halo/scaling tests for Phases 6 and 7 when host resources permit.

### Slow physics gate

```bash
conda run -n OpenONDA python -m pytest -q tests/fvm \
  -m "slow and not mpi and not openfoam"
```

Starting result: 14 pass and the fine-grid IBM drag test fails. The
implementation must produce no new failures. Run and report the complete gate;
do not hide the known failure.

For a clean green comparison of unaffected slow cases, additionally run the
same selection while deselecting only the exact known node:

```bash
conda run -n OpenONDA python -m pytest -q tests/fvm \
  -m "slow and not mpi and not openfoam" \
  --deselect='tests/fvm/test_ibm.py::test_ibm_square_force_and_wake_match_body_fitted_reference[0.125]'
```

### Required focused suites by phase

| Phase | Focused tests |
|---|---|
| 1 | empty/slip/symmetry BC, operator parity, one-step PIMPLE |
| 2 | Rhie--Chow consistency, pressure constraints, boundary schemes, PIMPLE |
| 3 | geometry, topology, cyclic, LSQ, mesh quality, mixed cells |
| 4 | momentum MMS, transient order, sources, linear telemetry |
| 5 | restart, diagnostics, async output, cache lifecycle |
| 6 | partition topology, coupling gather/scatter, MPI output/checkpoint |
| 7 | PETSc owned rows, nullspace, replicated/partitioned parity |
| 8 | turbulence validation, non-orthogonal characterization, full gates |

## 15. Performance acceptance dashboard

Maintain a small table in the implementation report after every phase:

| Metric | Baseline | Current | Change | Gate |
|---|---:|---:|---:|---|
| 100k initialization median | captured Phase 0 | | | no >5% regression |
| 100k warm step median | captured Phase 0 | | | no >5% regression |
| 100k peak RSS | captured Phase 0 | | | no >5% regression |
| BC projection kernel | captured Phase 0 | | | Phase 1 ≥20x |
| LSQ call count, cyclic serial | 2 | | | Phase 3 =1 |
| conductance calls/standard step | 4 | | | Phase 2 =2 |
| retained momentum matrices | 3 plus workspace | | | Phase 4 =1 |
| retained cache after close | unbounded | | | Phase 5 bounded |
| halo object collectives/step | multiple | | | Phase 6 =0 |
| PETSc Mat/KSP creations/step | 5 standard | | | Phase 7 =0 after init |
| 2-rank nontrivial warm step | captured Phase 0 | | | improve/no regression |
| 4-rank nontrivial warm step | captured Phase 0 | | | improve/no regression |

If a target is missed but the implementation is correct, do not invent a
speedup. Keep or revert the change based on measured time, memory, complexity,
and maintenance value, and document the decision.

## 16. Final definition of done

The implementing agent may declare the plan complete only when:

1. Every included phase is implemented or explicitly reported as blocked with
   evidence.
2. No excluded high-risk task has been folded into a performance refactor.
3. Fast serial, coupler-relevant, and collective MPI gates pass.
4. Slow physics has no new failures and the known IBM failure is reported
   unchanged.
5. Pyrefly has no new errors in changed files.
6. Ruff and compile checks pass.
7. Performance is measured from a clean source revision.
8. Before/after time, memory, iteration and cache-lifecycle results are
   recorded.
9. User-facing tutorials still specify only resources and scientific setup,
   not MPI/backend internals.
10. The completion report lists the remaining high-risk follow-ups:

    - IBM fine-grid force mismatch;
    - non-orthogonal correction certification;
    - true distributed dynamic Smagorinsky;
    - graph partitioning/distributed mesh input;
    - end-to-end GPU design;
    - numerically redesigned LSQ factorization.

Those follow-ups are intentionally not implementation work under this plan.
