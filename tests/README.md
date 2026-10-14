# Test suite index

The suite proves that the solvers compute the right physics and that an
installation works. Each test compares against an analytic result, an
independent reference, a conservation law or a complete solver run. Regression
checks for individual past defects are not kept.

| Location | Coverage |
| --- | --- |
| Root modules | Public API, documented examples, the installed-package check (`openonda-verify-install`), installers, MPI runtime, Numba cache, slip-slab images and ParaView-readable output. |
| [vpm/](vpm/) | Biot–Savart, treecode and FMM accuracy, kernel conformance, Runge–Kutta order, free-space energy, DVH/GBD diffusion, LES, slip-slab (planar) physics, Gaussian image mesh, VLM lifting line and restart. |
| [fvm/](fvm/) | Manufactured gradients and curl, PISO correctors, slip walls, mixed and non-orthogonal boundaries, time-step control, cyclic planar flow, mesher acceptance, MPI and restart, moving meshes (kinematics, free-stream preservation, Galilean invariance, oscillating-cylinder forces). |
| [coupler/](coupler/) | Transfer and interpolation accuracy, renewal conservation, interface iteration, coupled MPI launch, end-to-end coupled runs and restart, translating FVM meshes (frame invariance, restart, MPI). |
| [tutorials/](tutorials/) | Every tutorial setup reaches its solver, launchers clean, run and continue, and the tutorial style rule. |
| [support/](support/), [mesh_parity/](mesh_parity/) | Verification tools for tutorial results and the cfMesh reference; not collected as tests. |

Taichi, CUDA and the floating-point environment are process-global, so run
each file in its own process. [`run_isolated.py`](run_isolated.py) does this
and forwards pytest options:

```bash
python tests/run_isolated.py -m "not gpu and not slow"
python tests/run_isolated.py --junit-dir results -m qualification
python -m pytest tests/vpm/test_core_numerical_qualification.py
```

Markers are declared in [pyproject.toml](../pyproject.toml) and assigned in
[conftest.py](conftest.py): `unit`, `integration`, `qualification`,
`tutorial`, `slow`, `gpu` and `stochastic`. GPU tests need a CUDA device
(set `CUDA_PATH=/usr` for the Gaussian image mesh); MPI tests need `mpiexec`.
Numerical qualifications can write a report with
`--numerical-report=numerical-results.json`.

For FVM capability evidence, see the
[qualification reference](../docs/validation/fvm_qualification.md).
