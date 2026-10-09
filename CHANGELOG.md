# Changelog

All notable changes to OpenONDA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Native rewrite since 0.1.1: OpenONDA no longer wraps OpenFOAM and runs its own
finite-volume, vortex-particle and vortex-lattice solvers in Python.

### Breaking

- Import the solvers as `openonda.fvm`, `openonda.fvm.mesher`, `openonda.vpm`
  and `openonda.coupler`. The internal `source` package and
  `openonda.core.solvers` paths are gone.
- The immersed-boundary method is removed; FVM restart files written before
  this change do not resume.
- Python 3.11 is required.

### Added

- Native FVM solver with PISO, PIMPLE and SIMPLE pressure–velocity coupling,
  MPI partitioning, LES and a Cartesian mesher adapted from cfMesh.
- Native VPM solver on CPU and GPU (Taichi) with direct, treecode, FMM and
  Gaussian image-mesh induction, core spreading, random walk, DVH and GBD
  diffusion, LES and particle stabilization.
- Vortex-lattice wings and rotors coupled to the particle wake.
- FVM–VPM coupling of a resolved near field to a particle wake.
- Planar flow as one particle layer between free-slip planes, for every
  particle kernel and stabilization operator.
- Tutorials for vortex merging, vortex rings, wings, rotors, a quadcopter, the
  Taylor–Green vortex, boundary layers and coupled cylinder and cube flows.
- `openonda-verify-install`, which solves and restarts every solver.

### Changed

- The treecode defaults to opening angle θ=0.4 with third-order multipoles
  (was θ=0.1, first order): velocity and gradient errors of about 2e-4 and
  6e-4 relative to direct summation, at a fraction of the cost.
- VPM steps evaluate particle induction once less: an accepted state's field is
  reused by the next step's first stage when no particle changed in between.
- The treecode and FMM sort their Morton keys with an on-device radix sort, and
  the accepted-state limits are reduced on the device.

### Fixed

- PISO correctors update H/A with the previous correction's velocity change.
- Freestream and inletOutlet inflow faces carry flux in the first corrector.
- Slip and symmetry walls apply viscous stress to the wall-normal velocity only.

## [0.1.1] - 2025-03-31

Python interface to OpenFOAM coupled with a vortex-particle solver.

[Unreleased]: https://github.com/EngFlavioMartins/OpenONDA/compare/main...development
[0.1.1]: https://doi.org/10.5281/zenodo.15111460
