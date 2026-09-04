# Repository scripts

These are developer and maintainer tools for an OpenONDA source checkout. Run
them from the repository root after installing the development environment:

```bash
python -m pip install -e ".[dev]"
```

Installed users do not need this directory. Use `openonda info`, `openonda api`,
and `openonda tutorial` from any working directory instead.

## Installation helpers

- `install/install_conda.sh` creates or updates the reproducible `OpenONDA`
  Conda environment, installs the project, and verifies it outside the checkout.
  Use `--no-editable` for a fixed installation and `--parallel` for MPI/PETSc.
- `install/install_openvsp.sh` installs the optional OpenVSP application and its
  Python wrapper.
- `install/install_paraview.sh` installs the optional ParaView application used
  to inspect VTK output.

The OpenVSP and ParaView installers are optional; normal solver imports,
tutorial execution, and Matplotlib plots do not depend on them.

## Validation gates

```bash
python scripts/check_public_api.py
python scripts/check_api_completeness.py
python scripts/check_nomenclature.py --paths --generated
python scripts/validate_native_tutorials.py
```

The first three commands are static repository gates. The tutorial validator
runs a compact coupled FVM–VPM case through the installed public API; select an
explicit Taichi backend with `--compute-device` when needed.

## Benchmarks

```bash
python scripts/benchmarks/benchmark_fvm.py --output /tmp/fvm-benchmark.json
mpiexec -n 2 python scripts/benchmarks/benchmark_fvm_mpi.py \
  --cells-per-rank 512 --output /tmp/fvm-mpi-benchmark.json
python scripts/benchmarks/benchmark_vpm_step.py \
  --induction fmm --backend CPU --json /tmp/vpm-benchmark.json
python scripts/benchmarks/benchmark_panel_solver.py \
  --mode all --output /tmp/panel-benchmark.json
```

`benchmark_vpm_step.py` uses the production public VPM API and accepts `direct`,
`treecode`, or `fmm`. `bench_velocity_methods.py` is a lower-level diagnostic
for measuring the direct/treecode crossover and tree-build cost on a particular
Taichi device.

## Focused utilities

- `panel_mesh_audit.py` performs a panel-solver preflight audit on one STL file.

Every Python script supports `--help` when it defines command-line options. The
static API checks have no options and run directly as shown above.
