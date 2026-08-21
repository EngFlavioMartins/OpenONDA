<p align="center">
  <img src="./docs/logos/Logo_V7_Color.png" width="900" alt="OpenONDA" />
</p>

# OpenONDA

[![DOI](https://zenodo.org/badge/947793258.svg)](https://doi.org/10.5281/zenodo.15111460)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

OpenONDA is a native Python computational-fluid-dynamics library containing an
incompressible finite-volume method (FVM), a Taichi-accelerated vortex-particle
and vortex-lattice method (VPM/VLM), and an FVM↔VPM hybrid coupler. ONDA stands
for **Operator for Numerical Design and Aerodynamics**.

## Install

OpenONDA supports Python 3.11–3.13 on Linux x86-64 and macOS. Apple Silicon uses
current Taichi wheels; Intel macOS uses Python 3.11 with the last compatible
Taichi wheel.

Install the current development version in one command:

```bash
python -m pip install "OpenONDA @ https://github.com/EngFlavioMartins/OpenONDA/archive/refs/heads/development.zip"
```

Once a release is published on PyPI, the equivalent command is:

```bash
python -m pip install OpenONDA
```

No `PYTHONPATH`, repository-location export, or shell-startup modification is
needed. After installation this works from any directory:

```bash
cd /tmp
python -c "import openonda.fvm, openonda.vpm, openonda.coupler; print('OpenONDA ready')"
openonda-verify-install --require-site-packages
```

The verifier initializes Gmsh and Taichi and advances a small native FVM case;
it also rejects editable/source-tree imports when `--require-site-packages` is
used.

The default installation includes the serial FVM, VPM/VLM, coupler, internal
Gmsh meshing, PyAMG pressure solver, HDF5 output, and VTK visualization stack.

### Conda

For an isolated, reproducible environment on Linux or macOS:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
scripts/install/install_conda.sh
# Then run the two activation commands printed by the installer.
```

This installs OpenONDA in editable mode with the development tools, on the
Python version pinned in `scripts/environment/environment.yml` (currently
3.11). Edits under the clone take effect immediately, with no reinstall. Pass
`--no-editable` for a fixed copy in `site-packages`.

The installer reuses an existing Conda installation or offers to install
Miniforge. It does not edit `.bashrc`/`.zshrc`, auto-activate environments, or
require administrator privileges.

See [the installation guide](docs/installation.md) for supported platforms,
Intel/Apple-Silicon details, offline wheel installation, and troubleshooting.

For distributed FVM solves, install MPI, PETSc, mpi4py, and petsc4py from the
same Conda channel:

```bash
scripts/install/install_conda.sh --parallel
# Then run the two activation commands printed by the installer.
```

The serial installation is recommended unless the case explicitly selects MPI.
OpenVSP is optional and is only needed to regenerate geometry directly from
`.vsp3` files; cached DegenGeom input works without it.

## Use

The stable public modules are:

```python
from openonda import __version__
import openonda.fvm as fvm
import openonda.vpm as vpm
import openonda.coupler as coupling
```

Runnable cases live under `tutorials/FVM`, `tutorials/VPM`, and
`tutorials/coupled_FVM_VPM`. The coupled cases use the native FVM's internal
mesher:

```bash
cd tutorials/coupled_FVM_VPM/cube_flow
./allrun.sh
```

The cylinder-shedding and NACA 4412 workflows are in the same directory tree.
The installed public API, coupled subcycling, output layout, checkpoint names,
and restart parity can be exercised in an isolated temporary case with:

```bash
python scripts/validate_native_tutorials.py
python scripts/validate_native_tutorials.py --compute-device METAL  # Apple Silicon
```

### CPU and GPU execution

VPM selects a compatible Taichi backend automatically. To choose one explicitly:

```bash
OPENONDA_COMPUTE_DEVICE=CPU ./allrun.sh
OPENONDA_COMPUTE_DEVICE=METAL ./allrun.sh    # Apple Silicon
OPENONDA_COMPUTE_DEVICE=VULKAN ./allrun.sh   # Linux with a Vulkan driver
OPENONDA_COMPUTE_DEVICE=CUDA ./allrun.sh     # Linux with NVIDIA CUDA
```

`AUTO` selects Metal on macOS and a compatible GPU backend on Linux. Taichi
includes its runtime; a separate Vulkan SDK is not required.

## Develop and test

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
python -m pip install -e ".[dev]"
pytest tests/fvm -m "(unit or verification) and not slow and not mpi"
pytest tests/coupler -m "not mpi"
pyrefly check
```

See [docs/contributing.md](docs/contributing.md) for the architecture and code
quality requirements, and [docs/data_management.md](docs/data_management.md)
for the result-data policy.

## Citation

```bibtex
@software{openonda_zenodo,
  title  = {{OpenONDA}: Operator for Numerical Design and Aerodynamics},
  author = {Martins, Flavio A. C.},
  year   = {2025},
  doi    = {10.5281/zenodo.15111460},
  url    = {https://github.com/EngFlavioMartins/OpenONDA}
}
```

## License

OpenONDA is licensed under the GNU General Public License v3.0 or later. See
[license](license).

## Authors

- Flavio A. C. Martins — TU Delft, Faculty of Aerospace Engineering — [ORCID 0000-0002-1374-5760](https://orcid.org/0000-0002-1374-5760)
- Rention Pasolari — original 2D Python-wrapper contribution (pHyFlow)
