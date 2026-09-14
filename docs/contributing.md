# Contributing

Install a supported Python environment, then run:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -m "not qualification and not slow and not gpu"
```

Keep changes focused and preserve unrelated work in the checkout. Explain the
physical or numerical cause, the resulting behavior and the checks performed.
Do not relax reference values, tolerances or physical horizons to make a
scientific validation pass.

Follow the [code and experiment guidelines](development/code_guidelines.md)
when adding a solver feature or tutorial.

## Verification

Run appropriate tests from the [test guide](../tests/README.md), including
analytical or convergence qualifications when changing numerical algorithms.

```bash
ruff check source tests tutorials scripts openonda
ruff format --check source tests tutorials scripts openonda
python -m compileall -q source tests tutorials openonda
```

For FVM/coupler Python changes, also run `pyrefly check`. Taichi DSL code is
validated through runtime tests rather than ordinary static annotations.
For packaging changes, build both distributions with `python -m build`, install
the resulting wheel in a fresh virtual environment, change outside the
checkout, and run `python -I -m openonda.verify_install --require-site-packages`.
Check an editable installation separately. No Python path setup is required.

## Tutorials and output

Tutorials consume the installed `openonda.fvm`, `openonda.vpm` and
`openonda.coupler` interfaces. Input assets belong inside their case directory;
shared plotting support is `openonda.plotting`. Tutorial commands must work as
`python setup.py ...` and `python assets/name.py ...` after installation.
Scripts with relative imports register their local package using
`openonda.tutorial_runner.case_package` internally. Shell launchers use plain
`python`; Python subprocesses use `sys.executable`. Keep environment setup in
the installer and argument parsing in Python.

Do not commit generated solutions, backups, caches or build products as source.
Curated scientific reference data should include provenance and an explicit
reason for retention. Distribution archives exclude tutorial result trees.
Add tests for independent mathematical or behavioral contracts, combining
related edge cases; avoid snapshots of filenames, internal class layouts,
cosmetic logging, or fixed tutorial tuning constants.
