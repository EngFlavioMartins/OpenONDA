<p align="center">
  <img src="./docs/logos/Logo_V7_Color.png" width="720px"/>
</p>

# OFW: A Python Interface for In-Line Control of OpenFOAM Solvers

[![DOI](https://zenodo.org/badge/947793258.svg)](https://doi.org/10.5281/zenodo.15111460)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**OFW** (OpenFOAM-Wrapper) is a Cython/C++ extension that exposes OpenFOAM's
incompressible `pimpleFoam` solver to Python. It lets an external Python program
**advance an OpenFOAM case one time step at a time** and **read or write mesh
fields and boundary values between steps**: enabling in-line control, analytic
boundary conditions, and coupling of OpenFOAM with other scientific-computing
libraries (NumPy, SciPy).

This repository accompanies the *OpenFOAM Journal* (2026) article
*"OpenONDA: A Python Interface for In-Line Control of OpenFOAM Solvers"* and
contains the OFW solver together with five self-contained tutorial cases.

---

## Requirements

OFW is built and tested on Linux. It needs a working OpenFOAM installation (it
links against OpenFOAM's shared libraries) plus a small Python stack.

| Requirement        | Version (tested)        | Purpose                                   |
|--------------------|-------------------------|-------------------------------------------|
| OpenFOAM (OpenCFD) |  v2512       | Native solver libraries (sourced at build & run) |
| Python             | ≥ 3.10    | Host interpreter                          |
| Cython             | ≥ 0.29                  | Builds the extension (build-time only)    |         |
| mpi4py             | ≥ 3.1 *(optional)*      | Parallel (multi-rank) tutorial runs       |

OpenFOAM must be installed and its environment **sourced** before building or
running OFW, e.g.:

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
```

A reference install helper for OpenFOAM v2512 is provided in
[`scripts/install/install_openfoam.sh`](scripts/install/install_openfoam.sh).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
git checkout openfoam-journal
```

### 2. Create the Python environment

```bash
conda env create -f scripts/environment/environment.yml
conda activate ofw
```

Or install into an existing environment:

```bash
pip install -e ".[viz,mpi]"
```

### 3. Source OpenFOAM

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
```

### 4. Build the OFW extension

From the repository root, run the `Allwmake` script. It auto-detects/sources a
standard OpenFOAM install (if not already sourced) and compiles the Cython/C++
extension `ofw/fvm_solver*.so` in place:

```bash
./Allwmake            # build
./Allwmake --clean    # remove generated artefacts, then rebuild
```

Re-run `./Allwmake` whenever you change anything under `ofw/` or switch OpenFOAM
versions — a stale `fvm_solver*.so` silently runs outdated logic.

---

## Quick start

```python
from ofw import Solver
from ofw.utils import VortexRingFVM, set_boundary_conditions

solver = Solver(case_directory)          # initialise from an OpenFOAM case
coords = solver.get_cell_center_coordinates()
# ... impose an analytic initial/boundary condition, then march in time:
for _ in range(n_steps):
    solver.update_state()                # advance one FVM time step
```

The full list of interface methods, with usage notes, is in
[`docs/OFW_API.md`](docs/OFW_API.md).

---

## Tutorials

Each case under [`tutorials/`](tutorials/) is **self-contained** and ships a
pre-built mesh. Run a case end-to-end with its `Allrun` script:

```bash
cd tutorials/vortexRing
./Allrun        # decompose -> run (mpirun) -> reconstruct
./Allplot       # write diagnostics + field-slice figures into figures/
./Allclean      # reset the case
```

Useful environment knobs (read by `Allrun`):

- `NPROCS` — number of MPI ranks (default `4`; must match `system/decomposeParDict`).
- `OFW_STEPS` — override the step count for a short smoke run.

| Case                    | Demonstrates                                              |
|-------------------------|----------------------------------------------------------|
| `vortexRing`            | Time-varying Dirichlet BC; a self-advecting Lamb-Oseen vortex ring |
| `leapfroggingVortexRings` | Two coaxial rings leapfrogging down an elongated domain |
| `vortexFilament`        | Static analytic BC on an externally generated mesh       |
| `inflatingDipole`       | A time-dependent (growing-dipole) boundary condition     |
| `doubletFlow`           | Two independent sub-domains driven from one runner        |

---

## Repository layout

```
ofw/                 # the OFW package (Cython wrapper + C++ solver core + utils)
  foamSolverWrapper.pyx / .pxd     # Cython wrapper
  cpp/solver/                      # C++ PIMPLE solver core compiled into the extension
  utils/                           # analytic flow models + case-setup helpers
  setup.py                         # builds the fvm_solver extension
Allwmake             # builds the extension (parent-directory build script)
tutorials/           # five self-contained OpenFOAM cases
scripts/             # environment spec + OpenFOAM/conda/ParaView install helpers
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{OpenONDA_OFJ2026,
  title   = {{OpenONDA}: A Python Interface for In-Line Control of {OpenFOAM} Solvers},
  author  = {Martins, Flavio A. C. and van Zuijlen, Alexander and Ferreira, Carlos Sim\~{a}o},
  journal = {OpenFOAM Journal},
  year    = {2026},
  doi     = {TBD}
}

```

---

## License

OFW is licensed under the **GNU General Public License v3.0**; see [LICENSE](LICENSE).

OFW is derived from the **pHyFlow** OpenFOAM Python wrapper and links against
OpenFOAM; it is therefore distributed under the GNU GPL v3 as required by
OpenFOAM's license terms. The C++ solver core in `ofw/cpp/` is derived from
OpenFOAM source code and is likewise GPL-3.

---

## Authors & credits

- **Flavio A. C. Martins**; TU Delft, Faculty of Aerospace Engineering;
  [ORCID 0000-0002-1374-5760](https://orcid.org/0000-0002-1374-5760)
- **Artur Palha**, **Rention Pasolari**, **Lento Manickathan**; original
  pHyFlow OpenFOAM Python wrapper this work derives from.

---

## Use of AI

All code, algorithms, and documentation in this repository were originally written by humans (for 10+ years!). Documentation refinement, code cleanup (hygiene), and test generation were automated using agentic AI.
