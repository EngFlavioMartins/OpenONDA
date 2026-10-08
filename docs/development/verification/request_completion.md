# Current VPM restart and sampling checks

VPM checkpoints use format `10.3` and specified field names. Readers require
complete current numerical configuration, compute precision, particle fields,
accepted clocks, and stabilization transfer sums before changing solver state.
Removed configuration controls and older storage layouts are not supported.

Restart preserves the physical model exactly. CPU/GPU execution placement and
larger hard particle storage capacity may change without changing the operator.
The Gaussian finite-image operator supports `auto`, `cpu`, and `cupy_cuda`;
its mesh, interpolation, core, image, precision, and tail settings remain part
of the checked numerical configuration.

Fixed-plane sampling uses the bounded uniform grid and versioned frame metadata.
Continuation validates the declared geometry and exact stored coordinates before
changing sample indexes or histories. Frames without current metadata are rejected.

Executable restart checks live in `tests/vpm/test_start_latest.py` and
`tests/vpm/test_vlm_coupled_restart.py`.
