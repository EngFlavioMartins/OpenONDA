# Contributing

OpenONDA changes should preserve the scientific contracts, public namespaces,
and case-rooted output layout described in [nomenclature.md](nomenclature.md).
The complete naming migration is tracked in
[rename_project.md](rename_project.md) and
[rename_manifest.md](rename_manifest.md). Restartable state is called a
checkpoint; copied historical runs are called archives.

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
pytest -q tests
```

Run `pyrefly check` after changing Python under `source/solvers/fvm`,
or `source/coupler`. The Taichi-based VPM tree is excluded
from static type checking. Keep repository coverage to the four maintained
contracts; run case-specific scientific validation outside this test suite.

Run `python scripts/check_nomenclature.py` before publishing a serializer,
solver API, tutorial output, or checkpoint change. This is the repository gate
for newly introduced legacy physical-field identifiers; historical archives
and explicit migration adapters are intentionally excluded.

During the staged filesystem migration, also run
`python scripts/check_nomenclature.py --paths`; it reports legacy archive,
checkpoint, tutorial, and launcher names without modifying them.

Tutorial setup files use public namespace imports (`openonda.fvm as fvm`,
`openonda.vpm as vpm`, and `openonda.coupler as coupling`), uppercase physical
constants, a short usage docstring, and the same construction path in serial
and MPI.

## Pull requests

Explain the physical or numerical cause, the correction, and the commands that
verify it. Keep solver state, caches, build products, and local environment files
out of commits. Commit qualified tutorial `samples/` output for cross-device
post-processing; see [data_management.md](data_management.md).
