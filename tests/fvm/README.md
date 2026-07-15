# FVM test evidence

Run the serial reference suite in the canonical environment with:

```bash
conda run -n OpenONDA pytest -q tests/fvm tests/coupler/test_fvm_backend.py
```

Run the replicated PETSc collective checks separately so every test process is
launched by MPI:

```bash
conda run -n OpenONDA mpiexec -n 2 \
  /opt/anaconda3/envs/OpenONDA/bin/python -m pytest -q tests/fvm/test_petsc_parallel.py
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
