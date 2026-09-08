# Project tooling

Pip is the primary installation interface. These optional Conda files help
assemble a consistent interpreter and, where requested, an MPI/PETSc stack.
They do not repair Python imports or modify shell startup files.

| File | Purpose |
| --- | --- |
| `install/install_conda.sh` | Create/update a Conda environment, install OpenONDA and verify it outside the checkout. `--dev` selects editable development installation; `--parallel` selects MPI/PETSc. |
| `environment/environment.yml` | Base Python/numerical Conda environment; package metadata selects the compatible Taichi wheel. |
| `environment/environment-parallel.yml` | The base environment plus a coherent OpenMPI/PETSc stack. |

Use `bash scripts/install/install_conda.sh --help` for options. External
OpenVSP and ParaView installers have been replaced by
[upstream installation guidance](../docs/installation.md#optional-external-software).
