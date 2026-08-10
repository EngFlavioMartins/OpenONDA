# Contributing to OpenONDA

## Development installation

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
python -m pip install -e ".[dev]"
```

The editable install is for contributors only. User documentation uses a normal
wheel installation so imports behave independently of the checkout location.

Optional Git hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Architecture

```text
source/
├── solvers/
│   ├── FVM/          native incompressible finite-volume solver
│   └── VPM/          Taichi VPM and VLM solvers
├── coupler/          hybrid FVM↔VPM orchestration
├── utilities/        shared helpers
└── version.py
openonda/              stable public import facade
```

The solvers do not import each other's internals. Cross-solver orchestration
belongs in `source/coupler`; generally reusable code belongs in
`source/utilities`.

## Required checks

For Python changes under `source/solvers/FVM`, `source/coupler`, or
`source/utilities`, run Pyrefly and do not increase the existing error baseline:

```bash
pyrefly check
```

The VPM tree is excluded because Taichi kernel annotations are a runtime DSL.

Format and lint explicitly; pre-commit reports problems but does not rewrite
files:

```bash
ruff check --fix source tests
ruff format source tests
ruff check source tests
```

Run the blocking physics gates:

```bash
pytest tests/fvm -m "(unit or verification) and not slow and not mpi"
pytest tests/coupler -m "not mpi"
```

MPI/PETSc and slow physics-validation jobs run separately in CI.

## Taichi guidelines

- Reuse Taichi fields. Creating fields inside a time loop leaks device memory.
- Synchronize before Python reads results from asynchronous GPU kernels.
- Ensure concurrent writes use reductions or atomic operations.
- Call `Solver.reset_gpu()` before constructing an unrelated VPM solver in the
  same process; existing Taichi objects become invalid after reset.
- Keep CPU execution working because it is the portable CI and debugging path.

## Packaging changes

Runtime imports used by the normal FVM, VPM, or coupler paths belong in
`project.dependencies` in `pyproject.toml` and in the serial Conda environment.
Distributed MPI/PETSc dependencies also belong in
`scripts/environment/environment-parallel.yml`. After changing dependencies,
build and test the wheel outside the repository:

```bash
python -m build
python -m venv /tmp/openonda-wheel-test
/tmp/openonda-wheel-test/bin/python -m pip install dist/OpenONDA-*.whl
cd /tmp
/tmp/openonda-wheel-test/bin/python -c "import openonda.fvm, openonda.vpm, openonda.coupler"
```

## Getting help

- Open a [GitHub Discussion](https://github.com/EngFlavioMartins/OpenONDA/discussions).
- File an [issue](https://github.com/EngFlavioMartins/OpenONDA/issues).
