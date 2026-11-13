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
installing Miniforge into `~/miniforge3` first if Conda is absent; when Conda was
not active in the shell, it runs `conda init`, which adds Conda to `~/.bashrc`
(or `~/.zshrc`). The environment holds Python
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

OpenONDA is not yet published on PyPI or on a Conda channel; install it from a
clone as above.

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
`compute_device` selects a device with suitable drivers (`cuda`, `vulkan` or
`metal`; `auto` picks one: Metal on macOS, CUDA when `nvidia-smi` reports a GPU, Vulkan
otherwise, and names it in the log). Slip-slab cases, including planar flow, evaluate
their mirror images fastest with the optional CUDA extension, which supplies
CuPy and matching CUDA 12 libraries for an NVIDIA GPU:

```bash
pip install -e ".[gaussian-mesh-cuda12]"
```

The CUDA 12 runtime headers are found in the CuPy toolkit packages, in a Conda
environment, under `CUDA_PATH` or `CUDA_HOME`, or in a system toolkit such as
`/usr/local/cuda` or `/usr`. A slip-slab case whose particles run on CUDA
evaluates its images on the GPU too, and stops with a message if the extension
or the headers are missing. To evaluate the images on the CPU instead, pass
`gaussian_mesh_settings=vpm.GaussianSlabSettings(backend="cpu")` to
`SlipSlabInduction`. Particles on Vulkan, Metal or the CPU evaluate the images on
the CPU.

Check the GPU before a long run: `nvidia-smi` lists an NVIDIA card and its driver, and
`python -c "import taichi as ti; ti.init(arch=ti.cuda)"` starts Taichi on it.

## Check the installation

```bash
openonda info
openonda-verify-install
```

`openonda info` prints the installed version and location;
`openonda-verify-install` meshes, solves and restarts a tiny case with every
solver; it prints a JSON report and ends with "every installation check passed". `openonda api <name>` shows the documentation of any public class or
function, for example `openonda api openonda.vpm.ViscousConfig`.

## Troubleshooting

- `conda: command not found` right after the installation: open a new terminal, or run
  `source ~/miniforge3/etc/profile.d/conda.sh`, then `conda activate OpenONDA`.
- "Commands resolve outside the OpenONDA environment": another installation comes first on
  `PATH`; run `conda activate OpenONDA`.
- "mpiexec did not create the requested MPI communicator": mpi4py was built for another MPI
  than this `mpiexec`; re-run `source install.sh` to rebuild the environment.
- A slip-slab case stops with "Gaussian mesh requires …" while the particles run on CUDA:
  install the CUDA extension above, or evaluate the images on the CPU with
  `GaussianSlabSettings(backend="cpu")`.
- A particle case stops with "Taichi could not start compute_device=…": the requested GPU
  driver did not answer; check the GPU as above, or set `compute_device="cpu"` in the case.
- A particle case stops with "A Taichi runtime on 'cpu' was started outside OpenONDA": the
  script called `ti.init` itself before building the solver; remove that call, since the solver
  starts Taichi on the case's `compute_device`.
- Every start recompiles the Numba kernels and the log warns that the Numba cache is not
  writable: set `NUMBA_CACHE_DIR` to a writable directory.

## For maintainers: building release packages

A published GitHub release builds and tests the PyPI wheels
(`.github/workflows/release.yml`) and the Conda packages
(`.github/workflows/conda.yml`, recipes in `conda/recipes`). The Conda upload
needs the `ANACONDA_OWNER` repository variable (the anaconda.org channel, for
example `openonda`) and the `ANACONDA_API_TOKEN` secret, and the PyPI upload
needs trusted publishing configured for `release.yml`. Publish the
repackaged Taichi once per Taichi version by running the Conda workflow by
hand with `upload_taichi`.
