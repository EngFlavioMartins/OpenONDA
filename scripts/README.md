# Project tooling

Use `source install.sh` at the repository root. It creates and activates the
`OpenONDA` environment and installs the checkout editably with the solver and
plotting dependencies. Add `--dev` for the lint and packaging tools, and
`--gaussian-mesh-cuda12` for the optional CUDA Gaussian image mesh.

`environment/environment.yml` defines the shared Conda dependencies.
The helpers under `install/` are called by the root installer; they are not
separate user installation steps. See [installation](../docs/installation.md).


## Sharing the tutorial results with a plotting machine

Syncthing shares `tutorials/` itself, send-only, as the folder `openonda-tutorials`.
`tutorials/.stignore` (local, not tracked) leaves out the raw solver output: from
`solution/` only the run metadata, `coupler.log` and `coupler_diagnostics.jsonl` are
shared. Syncthing never follows symbolic links, so a case or run that lives under
`~/openonda_runs` behind a link arrives only as the link. On a Mac,
`tools/setup_sample_sync_mac.sh` in the thesis folder adds the receive-only side in
`~/OpenONDA_tutorials`, which the thesis and paper `allplot.sh` read.
