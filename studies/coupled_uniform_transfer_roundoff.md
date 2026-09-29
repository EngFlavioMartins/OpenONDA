# Uniform-flow coupling transfer

The 64-cell installed coupled smoke formerly failed its Gaussian-vorticity
divergence gate on the second transfer: the accepted FVM field had maximum
`|curl(u)| = 1.1086860364e-15 s^-1` and RMS curl component
`1.533934598e-16 s^-1`. The first transfer had exactly zero curl. The
dimensionless divergence metric divided two roundoff-scale quantities and
returned `0.1133477`, above its unchanged `0.08` acceptance limit.

The transfer now removes a **whole** FVM donor curl field only when every
cell is below `8 eps64 max|u| / h_min`, where `h_min` is the smallest distinct
donor-centre separation. This accounts for a stretched mesh without adding a
user control. If any cell has resolved curl, the complete field is retained.
The regular Gaussian-divergence gate still applies to mapped vorticity.

Verification on 2026-09-29:

- Native 64-cell coupled public factory, first and latest-resumed exchanges:
  `{'n_cells': 64, 'fvm_step': 4, 'vpm_step': 2}`; both transfers injected zero
  particles and mapped zero circulation. Log: `/tmp/installed_coupled_roundoff.log`.
- Ten focused unit cases cover stretched donor geometry, each velocity axis,
  speeds 0.1, 1, and 100 m/s, and a weak curl above the derived floor.
- Existing two-rank coupled native latest-restart test passed, including the
  transfer-geometry collective. Log: `/tmp/coupled_roundoff_mpi.log`.

The source test can be reproduced with
`python -m pytest -q tests/coupler/test_uniform_curl_roundoff.py`. The native
smoke is `openonda.verify_install._verify_native_coupled()`; the MPI test is
`tests/coupler/test_factory_mpi.py::test_partitioned_coupled_restart_keeps_boundary_collectives_in_order[latest]`.
