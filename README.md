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
from openonda.fvm import FVMSetup, Solver as FVMSolver, setup_fvm_solver
from openonda.vpm import VPMSetup, Solver as VPMSolver, setup_vpm_solver
from openonda.coupler import CouplerSetup, FVMVPMCoupler, setup_coupler
```

Runnable cases live under `tutorials/FVM`, `tutorials/VPM`, and
`tutorials/coupled_FVM_VPM`. The coupled cases use the native FVM's internal
mesher:

```bash
cd tutorials/coupled_FVM_VPM/cubeFlow
./allrun.sh
```

The cylinder-shedding and NACA 4412 workflows are in the same directory tree.
Standalone FVM, standalone VPM, and all three coupled installed-package smoke
cases can be exercised from isolated copies with:

```bash
scripts/validate_native_tutorials.sh
```

### CPU and GPU execution

VPM selects a compatible Taichi backend automatically. To choose one explicitly:

```bash
OPENONDA_PROCESSING_UNIT=CPU ./allrun.sh
OPENONDA_PROCESSING_UNIT=GPU_VULKAN ./allrun.sh   # Linux with a Vulkan driver
```

Taichi includes its runtime; a separate Vulkan SDK is not required. The helper
`scripts/install/install_vulkan_sdk.sh` only diagnoses whether a working Vulkan
driver is visible.

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
for the Git/DVC result-data policy.

## Citation

```bibtex
@software{openonda_zenodo,
  title  = {{OpenONDA}: Operator for Numerical Design and Fluidynamics},
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
