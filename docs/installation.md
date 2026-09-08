# Installation

OpenONDA is installed with pip. Native FVM, VPM and hybrid solvers do not
require OpenFOAM, a source checkout at runtime, or shell startup changes.
Python 3.11 is the common supported version across Linux x86-64, Apple Silicon,
and Intel macOS. Package metadata allows 3.11–3.13; Intel macOS uses the older
Taichi 1.7.1 wheel and requires 3.11. Windows is not currently qualified.

## Normal installation

From a cloned checkout, optionally create a virtual environment first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

After installation, change to any writable directory and run:

```bash
openonda info
python -m openonda.verify_install --require-site-packages
python -m pip check
openonda tutorial run fvm/taylor_green --workspace ./first-flow
```

The verifier checks installed resources, a rendered figure, CPU Taichi
initialization, Numba runtime compatibility and a real native FVM step. It creates temporary output and
returns a nonzero status on failure. `--require-site-packages` distinguishes a
normal installation from an editable checkout; omit it for editable installs.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

Run tests from the checkout using `python -m pytest`. Installed imports still
work from other directories; the editable installation tracks source changes.
For only the test runner, use `.[test]`. See [the test guide](../tests/README.md).

## Optional Python dependencies

| Extra | Purpose |
| --- | --- |
| `.[meshing]` | Gmsh API/import and accelerated STL reading through numpy-stl. Native meshers and the built-in STL reader work without it. |
| `.[parallel]` | mpi4py and petsc4py, requiring compatible MPI/PETSc system libraries. |
| `.[test]` | pytest. |
| `.[dev]` | Tests, lint/type/security tools, profiling and distribution building. |

NumPy, SciPy, Numba, Taichi and PyAMG supply the numerical runtime. HDF5,
VTK/PyVista, pandas and Matplotlib remain base dependencies because the current
solver construction, default output and introductory tutorials use them.
Making plotting optional requires first separating PyVista/VTK from the solver
import and output paths; declaring an extra alone would leave an incomplete
base installation. Documentation is Markdown, with no Sphinx build dependency.

## Optional external software

- **Gmsh:** install `.[meshing]` for its Python API. Linux may also require
  `libGLU` (`libglu1-mesa` on Debian/Ubuntu). See the
  [Gmsh installation manual](https://gmsh.info/doc/texinfo/#Installing-and-running-Gmsh-on-your-computer).
  Check it with `python -m openonda.verify_install --with-meshing`.
- **MPI/PETSc:** use one compatible MPI stack. The project provides
  `scripts/environment/environment-parallel.yml`; see the
  [PETSc installation guide](https://petsc.org/release/install/).
- **OpenVSP:** only needed to regenerate OpenVSP geometry; cached tutorial
  inputs are shipped. Install its API for a matching Python ABI from the
  [upstream distribution](https://openvsp.org/download.php). If that API lives
  in a separate environment, the optional rotor generation tool accepts
  `OPENONDA_OPENVSP_PYTHON` pointing to that interpreter. This selects an
  external executable; it does not change OpenONDA's import paths.
- **ParaView:** optional interactive visualization and some scene rendering.
  Obtain it from [ParaView](https://www.paraview.org/download/). Ordinary
  Matplotlib tutorial plots do not require the GUI.
- **OpenFOAM/cfMesh:** only needed for independent external parity studies or
  comparison data. Native solver operation and the introductory examples use
  neither program. Mesh-file interchange is distinct from running OpenFOAM.
- **LaTeX:** optional explicit publication rendering. The default plotting
  style uses Matplotlib's built-in math renderer.

## Conda alternative

The optional project helper creates a Conda environment and then installs the
same Python package:

```bash
bash scripts/install/install_conda.sh
```

Add `--dev` for an editable development installation or `--parallel` for the
MPI/PETSc environment. The helper can install Miniforge if Conda is absent;
use `--prompt` to confirm that download interactively. Pip is sufficient for
the normal installation and does not require this helper.
