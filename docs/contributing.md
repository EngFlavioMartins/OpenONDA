# Contributing

OpenONDA changes should preserve the scientific contracts, public namespaces,
and case-rooted output layout described in [nomenclature.md](nomenclature.md).

## Development setup

```bash
scripts/install/install_conda.sh
conda activate OpenONDA
python -m pip install -e ".[dev]"
```

Work on a focused branch and keep numerical changes separate from naming or
documentation cleanup. Do not change reference values, tolerances, mesh
resolution, or physical horizons merely to make a validation pass.

## Required checks

```bash
python -m compileall -q source tests tutorials openonda scripts
ruff check source tests tutorials scripts openonda
ruff format --check source tests tutorials scripts openonda
pytest -q tests/fvm -m "(unit or verification) and not slow and not mpi"
pytest -q tests/coupler -m "not mpi"
pytest -q tests/test_public_api_has_no_legacy_aliases.py tests/test_tutorial_contracts.py
```

Run `pyrefly check` after changing Python under `source/solvers/FVM`,
`source/coupler`, or `source/utilities`. The Taichi-based VPM tree is excluded
from static type checking. Add focused regression coverage for every bug fix
and run the nearest scientific validation before broad suites.

Tutorial setup files use public namespace imports (`openonda.fvm as fvm`,
`openonda.vpm as vpm`, and `openonda.coupler as coupling`), uppercase physical
constants, a short usage docstring, and the same construction path in serial
and MPI.

## Pull requests

Explain the physical or numerical cause, the correction, and the commands that
verify it. Keep generated simulations, caches, build products, and local
environment files out of commits; see [data_management.md](data_management.md).
