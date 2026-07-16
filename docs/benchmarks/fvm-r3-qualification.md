# Native FVM R3 qualification

**Date:** 2026-07-16

**Implementation commit:** `7fddc4caf`

**Supported contract:** `source/solvers/FVM/capabilities.json`

This record closes the production-completion and 3D-PIMPLE readiness plans.
It qualifies a deliberately bounded native solver, not every CFD model or
hardware combination.

## Qualified operating envelope

- static, constant-density, incompressible, Newtonian SIMPLE, PISO, and PIMPLE;
- serial NumPy/SciPy/PyAMG float64 execution;
- partitioned PETSc PIMPLE with owned cells, one halo layer, Gauss gradients,
  rank-local checkpoints, and PVTU output at one, two, and four ranks;
- first-order 3D tetrahedron, hexahedron, prism, pyramid, and polyhedral meshes
  accepted by the versioned contracts in `capabilities.json`;
- explicit boundary dispatch, pressure null-space handling, true residuals,
  sustained failure policies, and BDF1/BDF2 restart equivalence;
- fixed immersed bodies and one-rank native FVM--VPM coupling under their
  separate capability restrictions.

Unsupported combinations fail before stepping. These include dynamic meshes,
compressible or multiphase flow, moving immersed bodies, partitioned cyclic or
least-squares cases, partitioned FVM--VPM coupling, device FVM operators,
float32, and mixed precision.

## Release-gate disposition

| Gate | Disposition | Evidence |
|---|---|---|
| R1 correctness | Passed | analytical and manufactured convergence, square duct, published 3D cavity data, periodic 2D/3D flows, WALE decay, strict parser/mesh/boundary failures |
| R2 efficient CPU | Passed for the NumPy reference | in-place CSR values, preconditioner rebuild telemetry, buffered output, 1M-cell PIMPLE memory run |
| R3 distributed | Passed through four ranks | owned-row PETSc solves, halo invariance, global diagnostics and forces, complete restart, PVTU, weak-scaling report |
| R4 devices | Not advertised | Numba and Taichi CPU pass parity but deliver 0.98x and 0.57x of NumPy on the 10k case, below the required 1.5x acceleration gate |
| Fixed IBM | Qualified separately | transfer identities and two-level body-fitted force/wake comparison |
| Native FVM--VPM | Qualified reference only | one-rank conservation, Picard residual, and coupled restart tests |

The R4 work item is closed as a release target, not represented as successful
device acceleration. CUDA, Metal, and Vulkan remain unsupported until a future
implementation independently passes parity, memory, and end-to-end timing.

OpenFOAM installation and execution are not FVM release gates. The native mesh
reader retains its narrow ASCII compatibility contract, but CI and production
qualification do not require an OpenFOAM installation.

## Reproducible evidence

- `docs/benchmarks/fvm-serial-baseline-macos-arm64.json`
- `docs/benchmarks/fvm-backend-10k-macos-arm64.json`
- `docs/benchmarks/fvm-mpi-weak-scaling-macos-arm64.json`
- `tests/fvm/README.md`
- GitHub Actions run `29486075491`: Ubuntu canonical installation and its full
  native FVM/coupler suite passed; the follow-up workflow corrects repository
  installation in sharded jobs and four-rank runner oversubscription.

Local release commands completed successfully:

```bash
python -m pytest -q tests/fvm tests/coupler -m "not mpi and not openfoam"
mpiexec -n 2 python -m pytest -q tests/fvm/test_petsc_parallel.py
mpiexec -n 4 python -m pytest -q tests/fvm/test_petsc_parallel.py
ruff format --check source/solvers/FVM source/coupler tests/fvm tests/coupler tutorials/FVM
ruff check source/solvers/FVM source/coupler tests/fvm tests/coupler tutorials/FVM
mypy source/solvers/FVM
vulture source/solvers/FVM --min-confidence 80
bandit -q -r source/solvers/FVM
```
