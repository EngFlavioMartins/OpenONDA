<p align="center">
  <img src="./docs/logos/Logo_V7_Color.png" width="720px"/>
</p>

# OFW — A Python Interface for In-Line Control of OpenFOAM Solvers

[![DOI](https://zenodo.org/badge/947793258.svg)](https://doi.org/10.5281/zenodo.15111460)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**OFW** (OpenFOAM-Wrapper) is a Cython/C++ extension that exposes OpenFOAM's
incompressible `pimpleFoam` solver to Python. It lets an external Python program
**advance an OpenFOAM case one time step at a time** and **read or write mesh
fields and boundary values between steps** — enabling in-line control, analytic
boundary conditions, and coupling of OpenFOAM with other scientific-computing
libraries (NumPy, SciPy).

This repository accompanies the *OpenFOAM Journal* (2026) article
*"OpenONDA: A Python Interface for In-Line Control of OpenFOAM Solvers"* and
contains the OFW solver together with six self-contained tutorial cases.

---

## Requirements

OFW is built and tested on Linux. It needs a working OpenFOAM installation (it
links against OpenFOAM's shared libraries) plus a small Python stack.

| Requirement        | Version (tested)        | Purpose                                   |
|--------------------|-------------------------|-------------------------------------------|
| OpenFOAM (OpenCFD) | **v2506 / v2512**       | Native solver libraries (sourced at build & run) |
| C++ compiler       | C++17 (GCC/Clang)       | Compiles the extension (ships with OpenFOAM) |
| Python             | ≥ 3.10 (3.13 tested)    | Host interpreter                          |
| Cython             | ≥ 0.29                  | Builds the extension (build-time only)    |
| NumPy              | ≥ 1.24                  | Array exchange with the solver            |
| SciPy              | ≥ 1.10                  | Interpolation helpers                     |
| PyYAML             | ≥ 5.0                   | Reading case configuration               |
| matplotlib         | ≥ 3.6 *(optional)*      | Tutorial diagnostics plots                |
| PyVista + VTK      | ≥ 0.32 / ≥ 9.0 *(opt.)* | Tutorial field-slice rendering            |
| mpi4py             | ≥ 3.1 *(optional)*      | Parallel (multi-rank) tutorial runs       |

> **No Eigen or other third-party C++ libraries are required** — OpenFOAM itself
> is the only native dependency, and its version (**v2506** or **v2512**) must
> match the one sourced when building and running OFW.

OpenFOAM must be installed and its environment **sourced** before building or
running OFW, e.g.:

```bash
source /usr/lib/openfoam/openfoam2506/etc/bashrc
```

A reference install helper for OpenFOAM v2506 is provided in
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
source /usr/lib/openfoam/openfoam2506/etc/bashrc
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
| `vortexFilament`        | Static analytic BC on an externally generated mesh       |
| `parabolicAirfoilFlow`  | A complex analytic inflow profile imposed from Python    |
| `inflatingDipole`       | A time-dependent (growing-dipole) boundary condition     |
| `doubletFlow`           | Two independent sub-domains driven from one runner        |
| `LES`                   | Free-field run with a Smagorinsky LES turbulence model    |

The tutorials are the acceptance suite: each `Allrun` must complete and write
`solution/diagnostics.csv`.

---

## Repository layout

```
ofw/                 # the OFW package (Cython wrapper + C++ solver core + utils)
  foamSolverWrapper.pyx / .pxd     # Cython wrapper
  cpp/solver/                      # C++ PIMPLE solver core compiled into the extension
  utils/                           # analytic flow models + case-setup helpers
  setup.py                         # builds the fvm_solver extension
Allwmake             # builds the extension (parent-directory build script)
tutorials/           # six self-contained OpenFOAM cases
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

@software{ofw_zenodo,
  title   = {{OFW}: A Python Interface for In-Line Control of {OpenFOAM} Solvers},
  author  = {Martins, Flavio A. C.},
  year    = {2026},
  doi     = {10.5281/zenodo.15111460},
  url     = {https://github.com/EngFlavioMartins/OpenONDA}
}
```

---

## License

OFW is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE).

OFW is derived from the **pHyFlow** OpenFOAM Python wrapper and links against
OpenFOAM; it is therefore distributed under the GNU GPL v3 as required by
OpenFOAM's license terms. The C++ solver core in `ofw/cpp/` is derived from
OpenFOAM source code and is likewise GPL-3.

---

## Authors & credits

- **Flavio A. C. Martins** — TU Delft, Faculty of Aerospace Engineering —
  [ORCID 0000-0002-1374-5760](https://orcid.org/0000-0002-1374-5760)
- **Artur Palha**, **Rention Pasolari**, **Lento Manickathan** — original
  pHyFlow OpenFOAM Python wrapper this work derives from.
