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

Keep the checkout in place: installation is editable, so source changes take effect immediately. Re-run `source install.sh` to update the environment.

## First simulation

```bash
openonda tutorial run fvm/taylor_green --workspace ./first-flow
openonda tutorial plot fvm/taylor_green --workspace ./first-flow
```

This case follows viscous decay of a periodic vortex. Choose other flows from the [tutorial index](tutorials.md), or configure a case with the [FVM](fvm.md), [VPM/VLM](vpm.md), or [coupling](coupling.md) guide.

CPU execution is available with the standard installation. GPU cases require a device and drivers compatible with the selected `compute_device`. On Linux, the optional CUDA Gaussian slab installation supplies CuPy and its matched CUDA 12 libraries and headers; a compatible NVIDIA driver is required:

```bash
python install.py --gaussian-mesh-cuda12
```

Run this command in the activated environment. The CuPy [`ctk` extra](https://docs.cupy.dev/en/stable/install.html) supplies discoverable toolkit components without a tutorial-specific CUDA path. For that backend, `GaussianSlabSettings(backend="cpu")` selects CPU evaluation and `backend="cupy_cuda"` requires CUDA; the default `"auto"` permits CPU fallback. See [slab boundary conditions](coupling.md) before selecting this physical model.
