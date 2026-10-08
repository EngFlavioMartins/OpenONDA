# Installation

On Linux x86-64 or macOS (Intel or Apple Silicon), use Bash or Zsh:

```bash
git clone --depth 1 --branch development https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
source install.sh
```

The installer creates and activates the **OpenONDA** Conda environment, installs Python 3.11 and the solver, meshing, MPI, and plotting dependencies, and checks the installation. If Conda is absent, it installs Miniforge. Administrator access is not required.

In a new terminal:

```bash
conda activate OpenONDA
```

Keep the checkout in place: installation is editable, so edits to the code take effect immediately. Re-run `source install.sh` to update the environment, and add `--dev` to also install the test, lint and packaging tools. The installer adds Conda's start-up lines to your shell only when it had to install Conda itself.

After installation, `import openonda` works from any directory; cases need not live inside the checkout.

## Into an existing Python environment

With your own CPython 3.11 environment, install from the checkout with pip:

```bash
pip install -e .
```

Optional extras add features: `[parallel]` for MPI runs (mpi4py and petsc4py; Conda supplies a consistent MPI more reliably), `[meshing]` for Gmsh import, `[gaussian-mesh-cuda12]` for the CUDA image sums below, `[test]` or `[dev]` for the test suite and developer tools, for example `pip install -e ".[parallel,meshing]"`. Tutorial plots also need a LaTeX installation, which `source install.sh` provides.

## First simulation

```bash
cd tutorials/fvm/taylor_green
./allrun.sh
./allplot.sh
```

This case follows viscous decay of a periodic vortex. Choose other flows from the [tutorial index](tutorials.md), or configure a case with the [FVM](fvm.md), [VPM/VLM](vpm.md), or [coupling](coupling.md) guide.

## GPU acceleration

The standard installation runs on the CPU. Particle cases run on a GPU when `compute_device` selects a device with suitable drivers (`CUDA`, `VULKAN` or `METAL`; `AUTO` picks one). Slip-slab cases, including planar flow, evaluate their mirror images fastest with the optional CUDA extension, which supplies CuPy and matching CUDA 12 libraries for an NVIDIA GPU:

```bash
python install.py --gaussian-mesh-cuda12
```

Run it in the activated environment. Without it, these image sums fall back to a slower CPU evaluation.

## Check the installation

```bash
openonda info
```

prints the installed version and location; `openonda api <name>` shows the documentation of any public class or function, for example `openonda api openonda.vpm.ViscousConfig`.
