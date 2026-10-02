# Current VPM restart and sampling contracts

VPM checkpoints use format `10.2` and canonical field names. Readers require
complete current numerical configuration, compute precision, particle fields,
accepted clocks, and stabilization transfer ledgers before mutating solver state.
Removed configuration controls and older storage layouts are not admitted.

Restart preserves the physical model exactly. CPU/GPU execution placement and
larger hard particle storage capacity may change without changing the operator.
The Gaussian finite-image operator supports `auto`, `cpu`, and `cupy_cuda`;
its mesh, interpolation, core, image, precision, and tail settings remain part
of the authenticated numerical identity.

Fixed-plane sampling uses the bounded uniform grid and versioned frame metadata.
Continuation validates the declared geometry and exact stored coordinates before
changing sample indexes or histories. Frames without current metadata are rejected.

Focused executable checks live in `tests/vpm/test_backup_storage.py`,
`test_restart_config_changes.py`, `test_restart_config_changes_hdf5.py`,
`test_surface_sampling_bounds.py`, and `test_surface_sampling_resume.py`.
