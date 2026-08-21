# OpenONDA completion evidence

Durable verification record for the scientific, API, tutorial, and release audit
completed on 2026-08-21. The local implementation baseline was `bb469cb`; the
remote `development` baseline supplied with the audit was `18e7c27`.

## Repository and public API

- [x] Canonical modules and factories are the documented interface:
  `openonda.fvm`, `openonda.vpm`, `openonda.coupler`, and their
  `create_*` factories.
- [x] Serial and MPI cases use the same solver/coupler construction; user code
  contains no rank ownership branches.
- [x] Particle and physical names use `vortex_strength`, `core_radius`,
  `kinematic_viscosity`, `time_step_size`, `case_dir`, and
  `OPENONDA_COMPUTE_DEVICE` consistently.
- [x] Active `backup` vocabulary and legacy factories were removed. Compatibility
  reads remain only where old offline metadata must still be ingestible.
- [x] Solver, sampler, force CSV, checkpoint, and tutorial outputs are rooted in
  `<case_dir>/{solution,samples}`.
- [x] Public API and tutorial AST contracts are regression-tested by
  `tests/test_public_api_has_no_legacy_aliases.py` and
  `tests/test_tutorial_contracts.py`.

## FVM scientific certification

- [x] Fixed-time temporal study: Euler order `0.941`; BDF2 order `2.049`.
- [x] Periodic ABC flow: relative errors
  `[6.689e-4, 3.734e-4, 1.655e-4]`; observed spatial order `2.014`.
- [x] Taylor-Green vortex uses asymptotic levels `(16, 24, 32)`. At level 32,
  the central/upwind error ratio is `4.015`; central kinetic-energy relative
  error is about `-7e-6`, versus `-6.93e-3` for upwind.
- [x] Periodic transient pressure/velocity consistency: the cyclic-domain
  old-time face history is not double-counted by a boundary correction, and
  the correction is frozen once per physical PIMPLE step.
- [x] Explicit deviatoric/transpose stress is retained at physical boundaries;
  its constant-viscosity periodic form is reduced only where
  `grad(div(U)) = 0` analytically and the conservative face flux is the discrete
  divergence authority.
- [x] WALE decay refinement: level 12 peak dissipation `0.0085948` at `t=6.16`,
  level 16 `0.0094479` at `t=6.40`; maximum continuity defects
  `2.14e-12` and `8.15e-12`.
- [x] IBM loads include the fictitious-fluid momentum term
  `d/dt integral(V_body, U dV)`. The square-body test uses three grids,
  CFL-consistent time steps, a fixed physical horizon, and independent
  corrected-versus-raw force checks.
- [x] Full non-MPI FVM suite, including slow scientific tests:
  `python -m pytest -q tests/fvm -m "not mpi"`.

## VPM, VLM/panels, and coupling

- [x] VPM CPU core gate:
  `OPENONDA_COMPUTE_DEVICE=CPU python -m pytest -q tests/vpm -m "not gpu and not slow"`.
- [x] VLM loading, panel loads, particle absorption, checkpoint paths, and output
  placement have focused regression coverage.
- [x] Coupler threshold modes, interpolation, conservation, flux projection,
  pressure datum invariance, restart, and slow serial smoke tests pass. The
  closed-body pressure test is invariant to a `+17` datum shift.
- [x] Collective FVM MPI suite passes with two ranks (15 tests on each rank).
- [x] Coupled FVM-VPM public-API smoke passes with two ranks, two VPM steps, and
  three FVM substeps per VPM step.
- [x] Checkpoint filenames are generation-based and deterministic:
  `fvm_*.npz`, `vpm_*.h5/.xdmf`, `vpm_bc_*.npz`, and `manifest.json`.
- [x] Installed-API restart parity is exercised by
  `scripts/validate_native_tutorials.py` from an isolated working directory.

## Tutorial execution

- [x] Standalone Lamb-Oseen CPU smoke and a three-level grid-study driver run
  through the public API and write case-rooted samples.
- [x] Coupled cube smoke: one VPM step / three FVM substeps, exact checkpoint
  contract, finite fields, and case-rooted sample files.
- [x] Coupled NACA 4412 smoke: three VPM steps / twelve FVM substeps, finite
  fields, sampling, and generation-3 checkpoint contract.
- [x] Coupled cylinder smoke: one VPM step / five FVM substeps, finite fields,
  case-rooted IBM force history, and no `solution/samples` tree.
- [x] Production tutorial defaults remain unchanged; smoke reductions require an
  explicit flag or environment variable.

## Release gates

- [x] `python -m compileall -q source tests tutorials openonda scripts`.
- [x] Ruff check and format check over `source tests tutorials scripts openonda`.
- [x] Focused Pyrefly audit of the changed FVM solve/assembly paths: zero errors.
- [x] PEP 517 sdist and wheel build; both artifacts pass `twine check`.
- [x] Wheel installed outside the checkout in a temporary environment; native
  solve verifier passes with `--require-site-packages`; `pip check` reports no
  broken requirements.
- [x] Linux, Apple Silicon, and Intel macOS package coverage is represented in
  CI; Python support is documented as 3.11-3.13 with Intel pins for Taichi,
  Gmsh, and the last wheel-backed Numba/llvmlite line.
- [x] Nightly runs full FVM, two/four-rank PETSc, coupled two-rank MPI, canonical
  install, and isolated installed-API tutorial validation without masked test
  failures.

## Environment limitations

- Metal-parametrized VPM tests skip because the active Taichi build reports no
  Metal backend; CUDA was not available on this Apple Silicon host.
- OpenVSP import tests skip because its Python API is not available to the active
  environment. Native VLM/panel tests are green.
- Taichi cache-lock and sandbox OpenMPI socket warnings are environmental. MPI
  passed when launched with the active environment's explicit Python executable.
- Hosted GitHub Actions remains the authoritative cross-platform gate; local
  validation covers the native platform plus cross-platform wheel resolution.
