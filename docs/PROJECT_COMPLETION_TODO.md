# OpenONDA completion evidence

Durable verification record for the scientific, API, tutorial, and release
audit completed on 21 August 2026. The final repository audit started from
`development` commit `e93e0e8`.

## Repository and public API

- [x] The supported interfaces are `openonda.fvm`, `openonda.vpm`, and
  `openonda.coupler`, using their public factories and namespace imports.
- [x] Serial and MPI cases use the same solver/coupler construction; tutorial
  code contains no rank-ownership branches or private solver imports.
- [x] Particle and physical names consistently use `vortex_strength`,
  `core_radius`, `kinematic_viscosity`, `time_step_size`, `case_dir`, and
  `OPENONDA_COMPUTE_DEVICE`.
- [x] Solver, sampler, force, checkpoint, and tutorial outputs are rooted under
  `<case_dir>/{solution,samples}`.
- [x] Public API and tutorial contracts are pinned by
  `tests/test_public_api_has_no_legacy_aliases.py` and
  `tests/test_tutorial_contracts.py`.

## FVM scientific certification

- [x] Fixed-time temporal study: Euler order `0.941`; BDF2 order `2.049`.
- [x] Periodic ABC flow: relative errors
  `[6.689e-4, 3.734e-4, 1.655e-4]`; observed spatial order `2.014`.
- [x] Taylor--Green asymptotic comparison at level 32: central error
  `8.898657493336512e-05`, upwind error `0.003978594610177336`, and
  upwind/central ratio `44.71005444536532`. The central kinetic-energy relative
  error is `2.4639942229144566e-05`, compared with
  `-0.006903507495260594` for upwind.
- [x] WALE decay refinement: level 12 peak dissipation `0.0085948` at `t=6.16`,
  level 16 `0.0094479` at `t=6.40`; maximum continuity defects
  `2.14e-12` and `8.15e-12`.
- [x] IBM loads include the fictitious-fluid momentum term
  `d/dt integral(V_body, U dV)` and are covered by grid/time-step convergence
  and corrected-versus-raw force checks.

## VPM, VLM/panels, and coupling

- [x] CPU and Metal kernels cover induced velocity, gradients, stretching,
  viscous updates, and backend selection.
- [x] VLM loading, panel loads, particle absorption, checkpoint paths, and
  output placement have focused regression coverage.
- [x] Coupler threshold modes, interpolation, conservation, flux projection,
  pressure-datum invariance, restart, and serial/MPI smoke tests are covered.
- [x] Checkpoint filenames are generation-based and deterministic:
  `fvm_*.npz`, `vpm_*.h5/.xdmf`, `vpm_bc_*.npz`, and `manifest.json`.
- [x] Installed-API restart parity is exercised in an isolated directory by
  `scripts/validate_native_tutorials.py`.

## Tutorial audit

- [x] All 18 tutorial setup files use the compact style established by
  `tutorials/coupled_FVM_VPM/cube_flow/cubeFlow_setup.py`: short physical-case
  docstrings with a command, public namespace imports, uppercase constants,
  explicit derived quantities, compact sections, and a minimal entry point.
- [x] The cylinder reference case no longer overwrites its freestream velocity
  tuple with a scalar or references the undefined `U_INF` name; a one-step
  smoke run completed on 50,752 cells with finite output.
- [x] Taylor--Green ran for 10 CPU steps at `24^3`; final velocity relative
  error was `1.638436e-04`, energy error `7.037135e-05`, and continuity stayed
  near machine precision.
- [x] Lamb--Oseen ran one Metal step with 5,040 particles and finite VTK fields.
- [x] The coupled cube ran one Metal VPM step and three FVM substeps with 2,728
  FVM cells, 427 VPM particles, finite outputs, and complete restart artifacts.
- [x] The isolated native validator passed on Metal for two VPM steps and six
  FVM steps, including exact checkpoint/restart parity.

## Metal diagnosis

- [x] Host: Apple M4, 10 GPU cores, arm64 macOS, Metal supported.
- [x] Taichi 1.7.4 selects `arch=metal` outside the filesystem sandbox.
- [x] The earlier failure was caused by sandbox-denied Taichi cache locks and
  RHI memory mappings, not absent Metal hardware or an OpenONDA backend defect.
- [x] Validate locally with
  `python scripts/validate_native_tutorials.py --compute-device METAL`.

## Data and repository hygiene

- [x] Generated tutorial `solution/` trees remain local, while all 822 qualified
  sample files are versioned for cross-device post-processing, including sample
  output nested under grid studies.
- [x] Obsolete backend probes, legacy installers, one-off repair scripts,
  duplicate asset generators, task diaries, historical logo drafts, and stale
  VPM--LES intermediate experiments were removed.
- [x] VPM--LES conclusions are consolidated in `docs/vpm-les-validation.md`;
  large reproducible checkpoints are generated evidence rather than Git data.
- [x] `README.md`, installation, contribution, data-management, nomenclature,
  and agent guidance describe the current public API and compute-device names.

## Remaining scientific boundary

- The current variable-coefficient VPM eddy-viscosity operator is not
  solenoidal. It is not a validated production VPM--LES closure.
- DIAD is rejected for the tested particle architecture. Mansfield remains an
  unvalidated research route pending matched-Reynolds-number tests with adequate
  scale separation.
- OpenVSP import coverage still depends on installing its external Python API;
  native VLM and panel paths remain covered without it.

## Release gates

The release handoff requires compile, Ruff, tutorial-contract, focused
scientific regression, CPU/Metal native validation, Markdown-link, and Git diff
checks. Hosted GitHub Actions remains the authoritative cross-platform gate.
