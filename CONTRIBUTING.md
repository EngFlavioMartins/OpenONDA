# Contributing to OpenONDA

Thank you for helping. Questions, bug reports, new tutorials and fixes are all
welcome. Please follow the [code of conduct](CODE_OF_CONDUCT.md).

## Reporting a problem

Check [KNOWN_ISSUES.md](KNOWN_ISSUES.md) first; it may already describe the
problem and a workaround. Otherwise, open a [GitHub issue](https://github.com/EngFlavioMartins/OpenONDA/issues) and
include:

- the output of `openonda info` and `openonda-verify-install`;
- a short `setup.py` or script that reproduces the problem;
- what you expected, physically, and what you observed.

## Setting up for development

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
source install.sh --dev
```

`--dev` adds the test, lint and packaging tools to the `OpenONDA` Conda
environment and installs the checkout in editable mode.

## Running the tests

Taichi, CUDA and the floating-point environment are process-global, so run
each test file in its own process:

```bash
python tests/run_isolated.py -m "not gpu and not slow"
python -m pytest tests/fvm/test_manufactured_gradient_qualification.py
```

Tests prove solver physics against analytic or independent references,
complete runs, restart, MPI, installation and the public API; see the
[test guide](tests/README.md). Add a test when it checks a solver property.
Verify one-off fixes in a temporary case instead of adding a regression test
for the individual defect.

## Code style

- Run `ruff check` and `ruff format` before committing.
- Name quantities with plain CFD terms (`kinematic_pressure`,
  `volumetric_face_flux`, `blending_weight`); avoid abstract software jargon.
- Tutorials are physics-focused scripts: physical inputs, geometry, models,
  numerical choices and direct solver calls, briefly commented. Error handling,
  validation and restart logic belong in the solver, not in tutorials.
- Document SI units in docstrings and remove retired code instead of keeping
  compatibility branches.

## Pull requests

Branch from `development`, keep each pull request to one change, and describe
how you verified it. Contributions are licensed under the project's
[GPL-3.0-or-later](LICENSE) license.
