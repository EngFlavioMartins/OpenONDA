# Project tooling

Use `source install.sh` at the repository root. It creates and activates the
`OpenONDA` environment and installs the checkout editably with the solver and
plotting dependencies. Add `--dev` for the lint and packaging tools, and
`--gaussian-mesh-cuda12` for the optional CUDA Gaussian image mesh.

`environment/environment.yml` defines the shared Conda dependencies.
The helpers under `install/` are called by the root installer; they are not
separate user installation steps. See [installation](../docs/installation.md).


## Sharing tutorial samples with a plotting machine

`publish_samples.sh` copies the sampled results of the VPM, VLM-VPM and coupled
tutorials (every `samples/` directory, plus run metadata, `coupler.log` and
`coupler_diagnostics.jsonl` from `solution/`) into `~/Sync/openonda_samples`,
following symbolic links and keeping the tutorial layout. Field output, backups
and the FVM-only tutorials stay here. Syncthing shares that folder send-only.
Run `scripts/setup_sample_sync.sh` once after `sudo apt install syncthing`; it
starts Syncthing, adds the folder and a user timer that publishes every 15
minutes. Rerun it with the other machine's device ID to pair the two. On the
plotting machine, `allfetch.sh` at the root of the thesis and paper folders
copies what their figures need into their `data/` folders.
