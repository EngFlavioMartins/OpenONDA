# Installation

OpenONDA supports Python 3.11–3.13 on Linux x86-64 and Apple Silicon macOS.
Intel macOS is supported on Python 3.11: that combination uses Taichi 1.7.1
and Gmsh earlier than 4.13, plus Numba earlier than 0.63, so every compiled
dependency is available as an Intel binary wheel. Windows is not currently a
supported platform.

## Install with pip

Create and activate a virtual environment, then install OpenONDA:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "OpenONDA @ https://github.com/EngFlavioMartins/OpenONDA/archive/refs/heads/development.zip"
```

For a published release, replace the final command with
`python -m pip install OpenONDA`. Verify the installed package from outside the
source checkout:

```bash
cd /tmp
openonda-verify-install --require-site-packages
python -m pip check
```

The verifier imports the public modules, initializes Gmsh and Taichi, and
advances a small native FVM case. It also verifies the `py.typed` marker,
installed tutorial resources, and a headless Matplotlib render.
`--require-site-packages` detects an accidental import from a source checkout.

## Use the installed tutorials

Tutorials are distributed as immutable templates and copied to a user-owned
workspace before they run. No repository clone or manually configured import
path is required:

```bash
cd /tmp
openonda tutorial list
openonda tutorial run fvm/taylor_green
openonda tutorial create vpm/lamb_oseen_vortex ./lamb-oseen-study
openonda tutorial run vpm/lamb_oseen_vortex --workspace ./lamb-oseen-study
openonda tutorial plot vpm/lamb_oseen_vortex --workspace ./lamb-oseen-study
```

The first `run` command creates `./openonda-fvm-taylor-green` automatically.
Explicit `create` is useful when a stable workspace name is desired. Existing
cases are never overwritten. Solver output and plots are written only to the
workspace, not to the Python environment.

Inspect public API documentation from any directory with, for example,
`openonda api fvm.FVMSetup` or `python -m pydoc openonda.vpm`. The distribution
contains `py.typed`, allowing editors and language servers to use OpenONDA's
inline annotations without extra stub paths.

## Install with Conda

The repository installer creates a Python 3.11 environment and performs the
same runtime verification. It is non-interactive by default, discovers its
paths from the checkout and Conda environment, and installs Miniforge when
Conda is absent:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
scripts/install/install_conda.sh
```

It prints the two commands needed to activate the environment. The default is
an editable development installation; pass `--no-editable` for a fixed copy in
`site-packages`. Pass `--parallel` to use the MPI/PETSc environment. MPI,
PETSc, mpi4py, and petsc4py should come from the same Conda channel so their MPI
implementations agree. Pass `--prompt` only when an explicit confirmation is
desired before installing Miniforge.

## Select a VPM compute device

VPM uses Taichi and defaults to `AUTO`, which selects Metal on macOS and a
compatible GPU backend on Linux. Select a backend explicitly in the public
case configuration when needed:

```python
numerics = vpm.Numerics(compute_device="METAL")
```

Metal supports the normal `f32` VPM path. Use the CPU backend for `f64`, which
Metal does not support. To verify the installed coupled API and restart path on
Apple Silicon, run:

```bash
python scripts/validate_native_tutorials.py --compute-device METAL
```

## Offline installation

On a machine with network access and the same operating system, architecture,
and Python version, download OpenONDA and all binary dependencies into a wheel
directory:

```bash
python -m pip download --dest wheelhouse OpenONDA
```

Transfer `wheelhouse` to the offline machine, then install without consulting
a package index:

```bash
python -m pip install --no-index --find-links wheelhouse OpenONDA
```

If installing a development checkout instead of a published release, first
build its wheel with `python -m build --wheel`, place that wheel in
`wheelhouse`, and run the same offline install command.

## Troubleshooting

- Run commands with the environment's `python -m pip`; this avoids installing
  into a different Python interpreter.
- Do not set `PYTHONPATH` to the repository. A normal installation works from
  every directory without shell-startup changes.
- On Linux, an import error mentioning `libGLU.so.1` means the Gmsh runtime
  library is missing. On Debian/Ubuntu, install `libglu1-mesa`.
- On Intel macOS, use Python 3.11. Newer Python versions do not have a
  compatible Taichi wheel.
- On Apple Silicon, `METAL` is the native GPU backend. A failure inside a
  restricted container or sandbox does not establish that Metal is absent;
  repeat the probe from a normal terminal where Taichi can access its cache and
  the GPU device.
- GPU execution is optional. Set `compute_device="CPU"` in `vpm.Numerics` to
  use the portable Taichi CPU backend while diagnosing driver problems.
