# Installation

OpenONDA is installed with pip. Native FVM, VPM and hybrid solvers do not
require OpenFOAM, a source checkout at runtime, or shell startup changes.
Python 3.11 is the common supported version across Linux x86-64, Apple Silicon,
and Intel macOS. Package metadata allows 3.11–3.13; Intel macOS uses the older
Taichi 1.7.1 wheel and requires 3.11. Windows is not currently qualified.

The [README clone command](../README.md#installation) uses shallow Git history
and skips Git LFS downloads to reduce local storage. To retrieve the full
`development` history later, run `git fetch --unshallow origin`. Generated
`samples/` and `solution/` directories are not tracked; archive them separately
with their meshes, restart configuration and matching source revision.

## Normal installation

From a cloned checkout, optionally create a virtual environment first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python install.py
```

`install.py` installs the package and dependencies into that Python environment,
runs `pip check`, and verifies the installation from a temporary directory.
It requires no per-case interpreter variables or import-path configuration.
The equivalent package-only command is `python -m pip install .`.

After installation, change to any writable directory and run:

```bash
openonda info
python -m openonda.verify_install --require-site-packages
python -m pip check
openonda tutorial run fvm/taylor_green --workspace ./first-flow
```

The verifier checks installed resources, a rendered figure, a refined Cartesian
mesh using the compiled octree, CPU Taichi initialization, Numba runtime
compatibility and a real native FVM step. It creates temporary output and
returns a nonzero status on failure. `--require-site-packages` distinguishes a
normal installation from an editable checkout; omit it for editable installs.

The native Cartesian mesher automatically uses Numba for octree balancing.
This acceleration is part of the normal package: it requires no meshing extra,
external compiler, OpenFOAM/cfMesh installation, or repository-relative paths.
Numba compiles for the local CPU on first use and caches the result for later
processes, including runs launched from other working directories. The first
run after installation or a kernel update includes compilation overhead;
steady-state performance still depends on the CPU and mesh size.

Numba uses a writable package cache or falls back to its per-user cache for
read-only installations. Set `NUMBA_CACHE_DIR` to a writable persistent directory
if a shared installation requires an explicit cache location. Machine-specific
compiled cache files are not shipped in the wheel; each machine builds its own.
See [Numba's cache documentation](https://numba.readthedocs.io/en/stable/developer/caching.html).

## Development installation

```bash
python install.py --dev
```

Run tests from the checkout using `python -m pytest`. Installed imports still
work from other directories; the editable installation tracks source changes.
The equivalent package-only command is `python -m pip install -e ".[dev]"`.
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

Solver factories configure CPU use internally. MPI runs limit BLAS and Numba
to one thread per rank; `threadpoolctl` also updates libraries already loaded
by Python. Coupled VPM work uses the case CPU budget on its owning rank while
the other ranks wait. No per-tutorial thread exports are required. To construct
a coupled run, pass an `FVMSetup`, `VPMCase` and `CouplerSetup` to
`openonda.coupler.create_coupler` and use the returned driver as a context
manager; the factory handles MPI ownership and cleanup.

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
the normal installation and does not require this helper. The Conda helper
uses the same `install.py` once the environment exists. Activate the environment
once per terminal session, then run `python setup.py ...`, `python assets/name.py
...`, or `./allrun.sh` in a tutorial. The shell launchers use the active `python`.
