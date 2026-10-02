# Project tooling

Use `source install.sh` at the repository root. It creates and activates the
`OpenONDA` environment and installs the checkout editably, with development
and tutorial plotting tools. No installer options are needed.

`environment/environment.yml` defines the shared Conda dependencies.
The helpers under `install/` are called by the root installer; they are not
separate user installation steps. See [installation](../docs/installation.md).
