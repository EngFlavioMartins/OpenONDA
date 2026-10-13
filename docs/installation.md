# Installation

OpenONDA runs on Linux x86-64 and macOS (Intel or Apple Silicon) with
CPython 3.11, the newest Python for which Taichi publishes packages on every
supported platform. Windows is not supported.

## One command

```bash
git clone --depth 1 https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
source install.sh
```

The installer creates and activates an **OpenONDA** Conda environment,
installing Miniforge first if Conda is absent. The environment holds Python
3.11, the FVM, VPM and coupling dependencies, MPI with PETSc for parallel runs,
ParaView and LaTeX for the tutorial figures. OpenONDA is installed in editable
mode, so edits to the code take effect immediately. The installer finishes by
meshing, solving and restarting a tiny case with every solver. In a new
terminal, run `conda activate OpenONDA`; re-run `source install.sh` to update
the environment, and add `--dev` for the test, lint and packaging tools.

## Into your own Python environment

In an existing CPython 3.11 environment, from the cloned repository:

```bash
pip install -e .
```

Optional extras add features: `[parallel]` for MPI runs (mpi4py and petsc4py;
pip builds PETSc from source, so Conda is the easier route), `[meshing]` for
Gmsh import, `[gaussian-mesh-cuda12]` for the CUDA image sums below and `[dev]`
for the development tools, for example `pip install -e ".[parallel,meshing]"`.

## Released packages

A published release installs without cloning, from PyPI or the `openonda`
Conda channel:

```bash
pip install openonda
conda create -n openonda -c conda-forge -c openonda openonda
```

The Conda package brings MPI with PETSc; Taichi comes from the `openonda`
channel, which repackages its official wheels, and everything else from
conda-forge.

## First simulation

The tutorials are ready-to-run cases in the repository:

```bash
cd tutorials/fvm/taylor_green
./allrun.sh
./allplot.sh
```

This case follows viscous decay of a periodic vortex. Cases need not live
inside the repository: copy one anywhere and run it with the environment's
`python`. Choose other flows from the [tutorial index](tutorials.md), or
configure a case with the [FVM](fvm.md), [VPM/VLM](vpm.md), or
[coupling](coupling.md) guide.

Figures use LaTeX typography (Palatino) when `latex` and `dvipng` are on the
path, as in the one-command installation. Without them, Matplotlib's built-in
math text draws the same figures in a Palatino-like serif. The few scene
renders that compose TikZ overlays or ParaView views need `pdflatex` or
`pvpython`.

## GPU acceleration

The standard installation runs on the CPU. Particle cases run on a GPU when
`compute_device` selects a device with suitable drivers (`CUDA`, `VULKAN` or
`METAL`; `AUTO` picks one). Slip-slab cases, including planar flow, evaluate
their mirror images fastest with the optional CUDA extension, which supplies
CuPy and matching CUDA 12 libraries for an NVIDIA GPU:

```bash
pip install -e ".[gaussian-mesh-cuda12]"
```

The CUDA 12 runtime headers are found in the CuPy toolkit packages, in a Conda
environment, under `CUDA_PATH` or `CUDA_HOME`, or in a system toolkit such as
`/usr/local/cuda` or `/usr`. Without the extension, these image sums fall back
to a slower CPU evaluation.

## Check the installation

```bash
openonda info
openonda-verify-install
```

`openonda info` prints the installed version and location;
`openonda-verify-install` meshes, solves and restarts a tiny case with every
solver. `openonda api <name>` shows the documentation of any public class or
function, for example `openonda api openonda.vpm.ViscousConfig`.

## Release packages

A published GitHub release builds and tests the PyPI wheels
(`.github/workflows/release.yml`) and the Conda packages
(`.github/workflows/conda.yml`, recipes in `conda/recipes`). The Conda upload
needs the `ANACONDA_OWNER` repository variable (the anaconda.org channel, for
example `openonda`) and the `ANACONDA_API_TOKEN` secret, and the PyPI upload
needs trusted publishing configured for `release.yml`. Publish the
repackaged Taichi once per Taichi version by running the Conda workflow by
hand with `upload_taichi`.
