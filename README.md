# OpenONDA

**OpenONDA is a Python library for simulating fluid flow and aerodynamics.**
It combines finite-volume, vortex-particle, and vortex-lattice methods to study
viscous flows, wakes, wings, rotors, and interacting vortices.

## Main capabilities

- **FVM:** incompressible flow, native mesh generation, immersed bodies, and sampling.
- **VPM/VLM:** particle-based wake evolution and aerodynamic lifting surfaces, with CPU and supported GPU backends.
- **Hybrid coupling:** native FVM near-field flow coupled to a VPM outer domain.
- **Scientific workflows:** restartable simulations, diagnostics, and VTK/HDF5 output.

## Installation

Use Python **3.11–3.13** on Linux or Apple Silicon macOS; Intel macOS requires
Python **3.11**. A virtual environment is recommended.

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
python install.py
```

The installed package works from any directory:

```python
from openonda import fvm, vpm, coupler
```

The installer installs dependencies into the Python environment you are using
and verifies the result outside the checkout. Use `python install.py --dev`
to work on the source without reinstalling after edits. Ordinary
`python -m pip install .` is also supported.

Run a case directly with normal Python arguments:

```bash
cd tutorials/vpm/01_lamb_oseen_vortex
python setup.py vortex CS
python assets/rwm_ensemble.py vortex --number-of-realizations 10 --converge
```

See [installation details](docs/installation.md) for environment setup,
optional mesh import and MPI/PETSc support.

## Quick start

Run the small Taylor–Green vortex example from any directory:

```bash
openonda tutorial run fvm/taylor_green --workspace ./first-flow
```

This creates an editable example, runs the FVM solver, compares its decay with
the analytical solution, and saves a plot. Find the case under
`first-flow/tutorials/fvm/taylor_green/`, with results in its case-defined
`solution/` and `figures/` directories. No external CFD solver is needed.

## Tutorials

```bash
openonda tutorial list
openonda tutorial create vpm/vortex_ring ./ring-workspace
```

Each case includes its own instructions and input assets. The
[hybrid examples](docs/tutorials.md#catalog-and-scope) are another
quick introduction; wing, rotor, and vortex-interaction campaigns take longer.
See the [tutorial guide](docs/tutorials.md) for running and editing local cases.

## Documentation

- [FVM solver guide](docs/fvm.md)
- [VPM solver guide](docs/vpm.md)
- [FVM--VPM coupling guide](docs/coupling.md)
- [FVM package/API notes](source/solvers/fvm/README.md)
- [VPM numerical references](source/solvers/vpm/REFERENCES.md)
- [Tutorials and hybrid examples](docs/tutorials.md)
- [Installation and optional tools](docs/installation.md)
- [Contributor and AI-agent guidelines](AGENTS.md)
- Command-line help: `openonda --help` and `openonda api fvm.FVMCase`

## Development and contributing

Install with `python -m pip install -e ".[dev]"`, then follow the
[repository guidelines](AGENTS.md) and [test index](tests/README.md).
Report problems through [GitHub issues](https://github.com/EngFlavioMartins/OpenONDA/issues).

OpenONDA is licensed under [GPL-3.0-or-later](license).
For research use, see [citation.cff](citation.cff).
