# FVM performance, memory, and solver-overhaul audit

## Outcome

The native solver should retain its collocated PIMPLE algorithm,
but its storage and execution layer needs to become an LDU-oriented compiled
core. Replacing PIMPLE with a different pressure-velocity algorithm would make
validation harder without addressing the measured bottlenecks: sparse-object
duplication, Python process overhead, face-array traffic, PETSc setup, and the
retained VTK object graph.

The immediate repairs are already useful:

- Partitioned Gauss gradients now exchange complete owner-computed halo
  gradients. Previously, `linearUpwind` and corrected face operators consumed
  incomplete halo gradients at processor faces.
- `tolerance`, `relTol`, and final-stage tolerances use staged
  stopping semantics. Final PIMPLE iterations also select unrelaxed
  `UFinal`/`pFinal` behavior when no final relaxation factor is configured.
- Repeated pressure corrections reuse the unchanged discrete pressure matrix,
  while PETSc solver prefixes prevent pressure-PC options from changing the
  momentum solver.
- Mesh/CSR/global-index storage uses 32-bit indices where the mesh size permits
  it, diagnostic arrays are released before the next solve, and enstrophy is
  accumulated without retaining a vorticity field.
- Checkpoints are generation-atomic, capacity-checked, and restart-safe; JSONL
  diagnostics disable themselves cleanly on `ENOSPC` instead of terminating
  the flow solve.
- Every step records per-rank phase timings, Krylov setup/solve telemetry,
  aggregate current/peak RSS, and a deduplicated NumPy allocation inventory.

Identical-mesh native regressions measure force history and the full cell
velocity field. Comparisons between different meshes additionally measure
discretization error and cannot be expected to agree at machine precision.

## Enabling the profiler

The lightweight profiler is enabled by default and appends a structured record
per step to `solution/performance.jsonl` in either log mode. The phase table is
printed only in debug mode:

```bash
export FVM_LOG=debug                              # or LogConfig(mode="debug")
export FVM_PETSC_LOG=/absolute/path/to/petsc.log
```

`FVM_PETSC_LOG` writes PETSc's event summary when the solver closes. Set
`FVM_PROFILE=0` only when profiler collectives are inappropriate for a special
driver.

## Measured Pareto frontier

Measurements below use the 766,496-cell adaptive cube mesh on the development
laptop, f64, one BLAS/OpenMP thread per MPI rank, and a warmed second PIMPLE
step. They include all eight pressure solves and six component momentum solves
required by the case's two outer correctors, two pressure correctors, and one
non-orthogonal corrector.

| Configuration | MPI ranks | Step time | Process peak RSS | Decision |
|---|---:|---:|---:|---|
| Shared PETSc workspace, no GAMG cache | 4 | 8.335 s | 2.817 GiB | Low-memory fallback |
| Separate equation workspaces + cached GAMG | 4 | 6.112 s | 3.053 GiB | Production balance |
| Same optimized setup | 2 | 9.398 s | 2.390 GiB | Lowest measured RAM, too slow |
| Same optimized setup | 8 | 5.462 s | 4.284 GiB | Only 11% faster, poor RAM trade |
| Production setup with retained VTK exporter | 4 | about 6.3 s | 3.824 GiB observed | Current full-output run |

The previous 3.80-million-cell mesh needed about 6.15 GiB steady RSS and
9.29 GiB process high-water RSS, with a roughly 50 s warmed step. That is not a
safe production envelope on a 16 GiB workstation once the desktop and other
processes are included.

The four-rank optimized no-output stable NumPy inventory is:

| Allocation family | Aggregate size |
|---|---:|
| Geometry | 301.0 MiB |
| Mesh topology | 188.7 MiB |
| Solution fields and BDF/flux history | 130.4 MiB |
| Matrix/algorithm workspaces | 42.6 MiB |
| Derived diagnostic caches | 0 MiB after commit |
| **Deduplicated NumPy total** | **662.7 MiB** |

The remaining RSS is native PETSc/VTK storage, Python objects, allocator
retention, and four copies of imported extension/runtime state. With output
enabled, the retained PyVista/VTK grid accounts for approximately another
0.7--0.8 GiB on this mesh even though its native allocations are not visible to
the NumPy inventory.

## Time budget and realistic extraction

The warmed optimized four-rank step establishes the following current cost and
near-term floor. The floor is an engineering estimate for the existing
Python/PETSc architecture, not a claimed benchmark result.

| Module | Current critical path | Realistic floor | Main extraction |
|---|---:|---:|---|
| Momentum predictor | 2.62--2.9 s | 2.0--2.3 s | Fuse face passes; avoid repeated Python/SciPy object work; verify one matrix upload across component RHS solves |
| Pressure solve | 1.4--1.55 s | 1.1--1.3 s | Keep GAMG interpolation and coarse topology; tune rebuild criterion from measured convergence |
| Pressure assembly | 0.78--0.85 s | 0.50--0.65 s | Fill preallocated LDU/PETSc values directly; stop rebuilding generic CSR views |
| Velocity/flux correction | 0.54--0.58 s | 0.35--0.45 s | Fuse face interpolation, flux update, and cell correction |
| Turbulence + health diagnostics | about 0.52 s | 0.25--0.35 s | Reuse/fuse gradient reductions without retaining full derived fields |
| Continuity, logging, and gap | about 0.18 s | 0.10--0.15 s | Batch reductions and structured output |
| **Whole step** | **6.1--6.4 s** | **4.3--5.0 s** | Current architecture limit |

The practical memory floor without changing architecture is about 2.4--2.8
GiB peak for this mesh: two ranks reach the low end but sacrifice 54% step
time; four ranks with a shared workspace reach the useful low-memory end.
A direct streaming VTK writer should remove 0.6--0.8 GiB of retained production
RSS, but transient PETSc and face-array peaks remain.

## Recommended overhaul

### 1. Freeze a numerical oracle before changing representation

Keep the current implementation as the executable specification. For each
kernel and complete PIMPLE stage, compare assembled diagonal/off-diagonal/RHS,
corrected flux, residual trajectory, force decomposition, and final fields on
an identical mesh. Require serial/2/4-rank invariance and checkpoint continuation
parity. Use analytical and published benchmark comparisons as independent
references, with separate thresholds for identical-mesh solver error and
different-mesh discretization error.

### 2. Introduce a compact LDU mesh/operator store

Store owner, neighbour, lower/upper, and diagonal arrays as contiguous
structure-of-arrays with 32-bit labels. Keep only geometry required every
iteration; derive or stream rarely used geometry. Boundary patches should
reference slices instead of copied cell/face dictionaries. This removes much
of the generic SciPy CSR and Python-object graph while preserving the same
discrete equations.

### 3. Move hot face loops into one compiled extension

Use C++ (exposed through the project's existing native-extension toolchain) for
gradient, convection/diffusion assembly, Rhie--Chow pressure assembly,
correction, and force reduction. Fuse operations that traverse the same faces.
Numba/Taichi substitutions of individual kernels did not accelerate the large
case because memory traffic and Python/PETSc transitions dominate; a collection
of isolated JIT kernels is not the target architecture.

### 4. Give PETSc arrays directly, once

Preallocate the distributed sparsity pattern once and update numeric LDU/AIJ
values in place. Prefer a diag/off-diag AIJ representation or a tested
`MatShell` over materializing both SciPy CSR and PETSc matrices. Keep separate
pressure and momentum lifetimes selectable: cached pressure AMG for speed,
shared/destructive workspaces for low RAM. Solver residual reporting should
also use one consistent residual norm for serial and PETSc tolerances.

### 5. Replace retained PyVista topology in production output

Write appended-binary VTU/PVTU directly from 32-bit connectivity and stream
field buffers at the requested output precision. PyVista remains useful as a
post-processing dependency, but the solver should not retain a second native
mesh representation for its whole lifetime.

### 6. Targets and release gates

For the 766k-cell case, target 0.9--1.4 GiB steady and 1.5--2.0 GiB peak RSS,
with a 2.5--3.5 s four-rank step after the compiled/LDU phase. For the original
3.8M-cell reference mesh, target 3.5--5 GiB peak so it is viable on a 16 GiB
workstation. Treat these as design targets: publish measured numbers only after
the identical-mesh oracle, conservation gates, restart parity, and a complete
`t=20` cube run all pass.

## Decision

Do not replace PIMPLE, Rhie--Chow, or the finite-volume discretization strategy.
They now have direct benchmark parity evidence and changing them would reset the
trust baseline. Replace the generic Python/SciPy data path underneath them with
a compact LDU execution core, direct PETSc ownership, and streaming output.
That is the route to lower memory use while keeping the solver auditable from
Python.
