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
| OpenFOAM | v2406 (OpenCFD) |
| Python | >= 3.10 |
| Cython | >= 0.29 |
| NumPy | >= 1.24 |
| SciPy | >= 1.10 |
| Taichi | >= 1.7 (required for VPM) |
| GCC / Clang | compatible with your OpenFOAM installation |

> OpenFOAM must be installed and its environment sourced before using the OFW or FVM solvers.  
> Typical source command: `source /usr/lib/openfoam/openfoam2406/etc/bashrc`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
```

### 2. Create and activate a Python environment

```bash
conda create -n OpenONDA python=3.10
conda activate OpenONDA
pip install numpy scipy matplotlib cython taichi pydantic
```

Or install all Python dependencies at once:

```bash
pip install -e ".[full]"
```

### 3. Source OpenFOAM (required for FVM/OFW)

```bash
source /usr/lib/openfoam/openfoam2406/etc/bashrc
```

### 4. Build the C++ extensions and Cython wrapper

```bash
chmod +x Allwmake
./Allwmake
```

This script compiles the custom boundary conditions (`wmake`) and the Cython extension (`python setup.py build_ext --inplace`).

---

## Data Management

This repository uses **DVC (Data Version Control)** to manage large simulation data files separately from git.

### Quick Start

```bash
# Install DVC
pip install dvc

# Pull simulation data from remote storage
python -m dvc pull

# After running a simulation, track and backup the data
python -m dvc add tutorials/VPM/myCase/solution
python -m dvc push
git add tutorials/VPM/myCase/solution.dvc
git commit -m "Add myCase simulation"
git push
```

### Documentation

- **[Multi-Computer Workflow Guide](docs/multi_computer_workflow.md)** - Complete guide for managing data across multiple machines
- **[DVC Quick Reference](docs/DVC_QUICK_REFERENCE.md)** - Cheat sheet for common DVC operations
- **[DVC Workflow Details](docs/dvc_workflow.md)** - In-depth DVC usage guide

### What's Tracked Where?

- **Git**: Code, scripts, documentation, visualizations (PNG, PDF)
- **DVC**: Large simulation data (HDF5, VTK, mesh files)
- **Nextcloud**: Automatic cloud backup of DVC-tracked data

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
