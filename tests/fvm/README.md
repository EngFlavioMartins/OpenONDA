# FVM test evidence

Run the serial reference suite in the canonical environment with:

```bash
conda run -n OpenONDA pytest -q tests/fvm tests/coupler/test_fvm_backend.py
```

Run the PETSc checks separately at both qualified communicator sizes:

```bash
conda run -n OpenONDA mpiexec -n 2 \
  python -m pytest -q tests/fvm/test_petsc_parallel.py
conda run -n OpenONDA mpiexec -n 4 \
  python -m pytest -q tests/fvm/test_petsc_parallel.py
```

Unit tests cover operators, configuration, parsers, and failure contracts.
Verification tests compare analytical fields or refinement levels. Integration
tests advance the coupled pressure-velocity algorithm. Tests marked `mpi`
require a PETSc/mpi4py installation built against the launcher's MPI library.
Hardware-specific capability must be reported as a skip with its missing
dependency or device; a collected test must not silently select another backend.

`Solver.write_run_manifest()` records the revision, dirty state, dependency
versions, execution selection, configuration and mesh hashes, mesh provenance,
quality metrics, and host identity for verification and benchmark artifacts.

Every FVM test is assigned one primary marker during collection: `unit`,
`verification`, `integration`, or `parallel`. Run a class directly with, for
example, `pytest tests/fvm -m verification`. The CI workflow executes these
classes separately and preserves JUnit reports.

The scale benchmark records initialization, full-step, linear setup/solve, and
nonlinear/operator time with host and memory metadata:

```bash
conda run -n OpenONDA python scripts/benchmarks/benchmark_fvm.py \
  --sizes 10000 100000 --output artifacts/fvm-benchmark.json
```

For the declared 1M-cell memory qualification, the benchmark automatically
keeps the frozen direct-solver configuration through 100k cells and switches to
BiCGSTAB/AMG above that size:

```bash
conda run -n OpenONDA python scripts/benchmarks/benchmark_fvm.py \
  --sizes 1000000 --output artifacts/fvm-benchmark-1m.json
```

The 2026-07-15 macOS arm64 run completed one full 1M-cell PIMPLE step in
65.3 s with 3.62 GB peak RSS and a maximum continuity defect of
`1.11e-11`. The exact host and solver split are stored in
`docs/benchmarks/fvm-serial-baseline-macos-arm64.json`.

The partitioned weak-scaling benchmark is launched once per rank count:

```bash
mpiexec -n 4 python scripts/benchmarks/benchmark_fvm_mpi.py \
  --cells-per-rank 512 --output artifacts/fvm-mpi-4.json
```

The measured 1/2/4/8-rank report and four-rank support limit are stored in
`docs/benchmarks/fvm-mpi-weak-scaling-macos-arm64.json`. Backend timing and the
decision to treat Numba/Taichi CPU as parity-only are recorded in
`docs/benchmarks/fvm-backend-10k-macos-arm64.json`.
