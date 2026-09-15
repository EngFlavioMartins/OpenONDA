# Test suite index

Development rules, evidence requirements and verification commands are in
[AGENTS.md](../AGENTS.md#12-verification-commands-and-evidence).
This index describes where the current tests live.

| Location | Coverage |
| --- | --- |
| [vpm/](vpm/) | Particle methods, Taichi kernels, stage integration, induction, diffusion, turbulence, stabilization, boundary elements, health and restart/output. |
| [fvm/](fvm/) | Mesh/field validity, discretization, pressure–velocity solution, time-step control, boundaries and restart. |
| [coupler/](coupler/) | Transfers, interpolation, ownership, synchronization, conservation and coupled restart/failure behavior. |
| [tutorials/](tutorials/) | Case construction, installed entry points, samples, source selection and tutorial/figure behavior. |
| [mesh_parity/](mesh_parity/) | Native mesh checks and comparisons requiring their declared reference runtimes. |
| Root test modules | Public imports/API, installation, process runtime, storage and shared plotting. |

Select a test file or marker using normal pytest arguments:

```bash
python -m pytest tests/vpm/test_taichi_kernels.py
python -m pytest tests/tutorials/test_tutorial_style.py
python -m pytest tests -m "unit and not gpu"
```

Available markers are declared in [pyproject.toml](../pyproject.toml): `unit`,
`integration`, `qualification`, `tutorial`, `slow`, `gpu` and `stochastic`.
A marker describes the check; it does not certify the software's physical scope.

GPU checks require a supported device. MPI and external OpenFOAM/cfMesh
comparisons require their separate runtimes. These are not prerequisites for
native serial installation. [conftest.py](conftest.py) defines shared fixtures
and reporting options; numerical qualifications can write a report with
`--numerical-report=numerical-results.json`.

For FVM capability-specific evidence, see the
[qualification reference](../docs/validation/fvm_qualification.md).
