# Project tooling

Use `source install.sh` at the repository root. It creates and activates the
`OpenONDA` environment and installs the checkout editably with the solver and
plotting dependencies. Add `--dev` for the test, lint and packaging tools, and
`--gaussian-mesh-cuda12` for the optional CUDA Gaussian image mesh.

`environment/environment.yml` defines the shared Conda dependencies.
The helpers under `install/` are called by the root installer; they are not
separate user installation steps. See [installation](../docs/installation.md).

Maintainers can repeat a fresh Ubuntu installation using the
[VirtualBox baseline](../docs/development/verification/clean_install_vm.md).
