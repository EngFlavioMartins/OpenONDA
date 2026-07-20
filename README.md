<p align="center">
  <img src="./docs/logos/Logo_V7_Color.png" width="900px"/>
</p>

# OpenONDA — Hybrid VPM-FVM Solver with Python Interface

[![DOI](https://zenodo.org/badge/947793258.svg)](https://doi.org/10.5281/zenodo.15111460)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

OpenONDA is a Computational Fluid Dynamics (CFD) framework that integrates a GPU-accelerated Vortex Particle Method (VPM), a pure-Python Finite Volume Method (FVM), and an OpenFOAM-Python interface (OFW) under a unified Python API.

ONDA stands for **"Operator for Numerical Design and Aerodynamics"**.

---

## Requirements

| Requirement | Version tested |
|---|---|
| OpenFOAM | optional; OFW backend only |
| Python | 3.13 |
| Cython | >= 0.29 |
| NumPy | >= 1.24 |
| SciPy | >= 1.10 |
| Taichi | 1.7.4 (required for VPM) |
| GCC / Clang | compatible with your OpenFOAM installation |

> The native Python FVM and VPM solvers do not require OpenFOAM. The separate
> OFW backend requires a supported Ubuntu/OpenFOAM installation.

Helper install scripts for a fresh machine live in [`scripts/install/`](scripts/install/):
`install_anaconda.sh`, `install_openfoam.sh` (optional OFW dependency),
`install_vulkan_sdk.sh` (GPU backend for VPM) and `install_paraview.sh`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
```

### 2. Create and activate a Python environment

```bash
conda env create -f scripts/environment/environment.yml
conda activate OpenONDA
```

The FVM–VPM cube tutorial also accepts an existing `OpenONDA-VPM` environment.
To update that named environment with the distributed FVM dependencies, run
`OPENONDA_CONDA_ENV=OpenONDA-VPM scripts/install/install_anaconda.sh`.

Or install all Python dependencies at once into an existing env:

```bash
pip install -e ".[full]"
```

### 3. Optional: source OpenFOAM for OFW

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
```

### 4. Optional: build the OFW extension

```bash
scripts/install/build_solvers.sh
```

This (re)compiles the **OFW** solver — the Cython/C++ extension
(`source/solvers/OFW/fvm_solver*.so`) that links against OpenFOAM. It is the
only compiled component. The native **FVM** solver and **FVM-VPM coupler** are
pure Python; **VPM** uses Python and Taichi JIT kernels.

Re-run this script whenever you change anything under `source/solvers/OFW/`,
switch OpenFOAM versions, or pull changes that touch the OFW sources — a stale
`fvm_solver*.so` silently runs outdated logic. Use `--clean` to force a full
rebuild from scratch:

```bash
scripts/install/build_solvers.sh --clean
```

---

## Data Management

This repository keeps **code in git** and **large simulation data in DVC**, with
**Nextcloud** as the DVC remote so results sync across machines.

### What's tracked where?

- **Git** — files needed to _run_ a case: setup scripts, `system/`,
  `constant/{transportProperties,turbulenceProperties}`, `constant/polyMesh.orig/`,
  `0.orig/`, `assets/`, and visualizations (`figures/*.png`, `*.pdf`).
- **DVC** — files _produced_ by a run: `solution/`, `referenceFlow/`, and
  OpenFOAM reconstructed time directories.
- **Ignored** — regenerable scratch: `processor*/`, runtime `constant/polyMesh/`
  and `0/`, `*.foam`, `log.*`, `VTK/`, `postProcessing/`.

### Quick start

```bash
# Start a session
git pull && dvc pull

# After running a simulation: track all new results, then back them up
scripts/dvc_add_solutions.sh        # dvc add's solution/, referenceFlow/, time dirs
dvc push                            # upload to Nextcloud
git commit -am "Add myCase results" && git push
```

See **[docs/data_management.md](docs/data_management.md)** for the full folder
policy, multi-computer workflow, and troubleshooting.

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

@software{openonda_zenodo,
  title   = {{OpenONDA}: Operator for Numerical Design and Fluidynamics},
  author  = {Martins, Flavio A. C.},
  year    = {2025},
  doi     = {10.5281/zenodo.15111460},
  url     = {https://github.com/EngFlavioMartins/OpenONDA}
}
```

---

## License

OpenONDA is licensed under the **GNU General Public License v3.0**.
See [license](license) for details.

The custom boundary conditions in `source/solvers/OFW/cpp/` are derived from OpenFOAM source code and are subject to the **GNU General Public License v3.0** as required by OpenFOAM's license terms.

---

## Authors

- **Flavio A. C. Martins** — TU Delft, Faculty of Aerospace Engineering — [ORCID 0000-0002-1374-5760](https://orcid.org/0000-0002-1374-5760)
- **Rention Pasolari** — original 2D OpenFOAM Python wrapper (pHyFlow)
