# Installing OpenONDA

OpenONDA is a normal Python distribution: after installation, `openonda` is
imported from the environment's `site-packages` directory. Do not set
`PYTHONPATH`, add the repository to `sys.path`, or modify a shell startup file.

## Recommended: pip

Use Python 3.11, 3.12, or 3.13 on Linux or macOS, preferably in a virtual
environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "OpenONDA @ https://github.com/EngFlavioMartins/OpenONDA/archive/refs/heads/development.zip"
```

When a release is available on PyPI, replace the last command with
`python -m pip install OpenONDA`. The default distribution contains the native
FVM, VPM/VLM, FVM–VPM coupler, Gmsh and Cartesian mesh support, Taichi kernels,
HDF5 output, PyAMG, NumPy-STL geometry support, and VTK/PyVista visualization.
Git checkout metadata is recorded in run manifests when the optional developer
dependency `pygit2` is installed; normal installed-package runs do not require
Git or a repository checkout.

Verify from outside the checkout:

```bash
cd /tmp
python -c "import openonda.fvm, openonda.vpm, openonda.coupler; print('OpenONDA ready')"
openonda-verify-install --require-site-packages
python -m pip check
```

The verifier initializes Gmsh and Taichi and advances a small native FVM
problem. With `--require-site-packages`, it also fails if Python resolves an
editable checkout instead of the installed distribution.

To exercise representative standalone FVM and VPM tutorials plus all three
native coupled workflows from isolated copies, run:

```bash
scripts/validate_native_tutorials.sh
```

## Conda/Miniforge

The repository installer is the reproducible option for a complete scientific
environment:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
scripts/install/install_conda.sh
```

It creates or updates `OpenONDA` from `scripts/environment/environment.yml`,
installs OpenONDA into that environment **in editable mode with the development
tools**, initializes Gmsh and Taichi, advances a native FVM step from `/tmp`,
and runs `pip check`. Editable means the environment links back to your clone,
so edits under the repository take effect immediately with no reinstall.

Use `--name NAME` to choose another environment, `--no-editable` to install a
fixed copy into `site-packages` instead, or `--parallel` for a single-channel
OpenMPI/PETSc/mpi4py/petsc4py stack.

The installer refuses to reuse an environment whose Python differs from the
version pinned in the environment file, so a stale environment cannot silently
be reused.

The environment uses Python 3.11 so one definition works on Linux, Apple
Silicon, and Intel macOS. Platform markers in `pyproject.toml` select current
Taichi/Gmsh wheels where available and compatible pinned wheels on Intel macOS.

## Wheels and offline machines

Build artifacts on a connected machine with:

```bash
python -m pip install build
python -m build
```

Copy the wheel plus downloaded dependency wheels to the target, then install
from that directory with `python -m pip install --no-index --find-links .
OpenONDA-*.whl`.

## Optional components

Distributed FVM execution uses the Conda `--parallel` environment. OpenVSP is
only needed to regenerate geometry directly from `.vsp3` models; cached
DegenGeom input and all other solver paths work without it. ParaView is also
optional because PyVista/VTK output support is installed by default.

## Troubleshooting

- `ModuleNotFoundError: openonda` means OpenONDA was not installed into the
  Python interpreter currently running. Compare `which python` and
  `python -m pip --version`, then install with that same `python -m pip`.
- Never fix this error by exporting `PYTHONPATH`; doing so hides environment
  mistakes and makes tutorials depend on a checkout location.
- Set `OPENONDA_PROCESSING_UNIT=CPU` for portable VPM diagnostics when a GPU
  backend or driver is unavailable.
